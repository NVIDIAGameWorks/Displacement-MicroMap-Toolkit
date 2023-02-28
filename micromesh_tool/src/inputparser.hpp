/*
* SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* SPDX-License-Identifier: LicenseRef-NvidiaProprietary
*
* NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
* property and proprietary rights in and to this material, related
* documentation and any modifications thereto. Any use, reproduction,
* disclosure or distribution of this material and related documentation
* without an express license agreement from NVIDIA CORPORATION or
* its affiliates is strictly prohibited.
*/

#pragma once
#include <iostream>
#include <string>
#include <variant>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <map>


static constexpr int MAX_LINE_WIDTH = 60;

//--------------------------------------------------------------------------------------------------
// Command line parser.
//  std::string inFilename = "";
//  bool printHelp = false;
//  CommandLineParser args("Test Parser");
//  args.addArgument({"-f", "--filename"}, &inFilename, "Input filname");
//  args.addArgument({"-h", "--help"}, &printHelp, "Print Help");
//  bool result = args.parse(argc, argv);
//
class CommandLineParser
{
public:
  // These are the possible variables the options may point to. Bool and
  // std::string are handled in a special way, all other values are parsed
  // with a std::stringstream. This std::variant can be easily extended if
  // the stream operator>> is overloaded. If not, you have to add a special
  // case to the parse() method.
  using Value = std::variant<int32_t*, uint32_t*, double*, float*, bool*, std::string*>;

  // The description is printed as part of the help message.
  CommandLineParser(const std::string& description)
      : m_description(description)
  {
  }

  // Adds a possible option. A typical call would be like this:
  // bool printHelp = false;
  // cmd.addArgument({"--help", "-h"}, &printHelp, "Print this help message");
  // Then, after parse() has been called, printHelp will be true if the user
  // provided the flag.
  void addArgument(std::vector<std::string> const& flags, Value const& value, std::string const& help)
  {
    m_arguments.emplace_back(Argument{flags, value, help});
  }

  // Prints the description given to the constructor and the help for each option.
  void printHelp(std::ostream& os = std::cout) const
  {
    // Print the general description.
    os << m_description << std::endl;

    // Find the argument with the longest combined flag length (in order to align the help messages).
    uint32_t maxFlagLength = 0;
    for(auto const& argument : m_arguments)
    {
      uint32_t flagLength = 0;
      for(auto const& flag : argument.m_flags)
      {
        // Plus comma and space.
        flagLength += static_cast<uint32_t>(flag.size()) + 2;
      }

      maxFlagLength = std::max(maxFlagLength, flagLength);
    }

    // Now print each argument.
    for(auto const& argument : m_arguments)
    {
      std::string flags;
      for(auto const& flag : argument.m_flags)
      {
        flags += flag + ", ";
      }

      // Remove last comma and space and add padding according to the longest flags in order to align the help messages.
      std::stringstream sstr;
      sstr << std::left << std::setw(maxFlagLength) << flags.substr(0, flags.size() - 2);

      // Print the help for each argument. This is a bit more involved since we do line wrapping for long descriptions.
      size_t spacePos  = 0;
      size_t lineWidth = 0;
      while(spacePos != std::string::npos)
      {
        size_t nextspacePos = argument.m_help.find_first_of(' ', spacePos + 1);
        sstr << argument.m_help.substr(spacePos, nextspacePos - spacePos);
        lineWidth += nextspacePos - spacePos;
        spacePos = nextspacePos;

        if(lineWidth > MAX_LINE_WIDTH)
        {
          os << sstr.str() << std::endl;
          sstr = std::stringstream();
          sstr << std::left << std::setw(maxFlagLength - 1) << " ";
          lineWidth = 0;
        }
      }
    }
  }


  // The command line arguments are traversed from start to end. That means,
  // if an option is set multiple times, the last will be the one which is
  // finally used. This call will throw a std::runtime_error if a value is
  // missing for a given option. Unknown flags will cause a warning on
  // std::cerr.
  bool parse(int argc, char* argv[], std::ostream& os = std::cerr)
  {
    bool result = true;

    // Skip the first argument (name of the program).
    int i = 1;
    while(i < argc)
    {
      // First we have to identify whether the value is separated by a space or a '='.
      std::string flag(argv[i]);
      std::string value;
      bool        valueIsSeparate = false;

      // If there is an '=' in the flag, the part after the '=' is actually
      // the value.
      size_t equalPos = flag.find('=');
      if(equalPos != std::string::npos)
      {
        value = flag.substr(equalPos + 1);
        flag  = flag.substr(0, equalPos);
      }
      // Else the following argument is the value.
      else if(i + 1 < argc)
      {
        value           = argv[i + 1];
        valueIsSeparate = true;
      }

      // Search for an argument with the provided flag.
      bool foundArgument = false;

      for(auto const& argument : m_arguments)
      {
        if(std::find(argument.m_flags.begin(), argument.m_flags.end(), flag) != std::end(argument.m_flags))
        {

          foundArgument = true;

          // In the case of booleans, the value is not needed.
          if(std::holds_alternative<bool*>(argument.m_value))
          {
            if(!value.empty() && value != "true" && value != "false")
            {
              valueIsSeparate = false;  // No value
            }
            *std::get<bool*>(argument.m_value) = (value != "false");
          }
          // In all other cases there must be a value.
          else if(value.empty())
          {
            os << "Failed to parse command line arguments. Missing value for argument " << flag << std::endl;
            return false;
          }
          // For a std::string, we take the entire value.
          else if(std::holds_alternative<std::string*>(argument.m_value))
          {
            *std::get<std::string*>(argument.m_value) = value;
          }
          // In all other cases we use a std::stringstream to convert the value.
          else
          {
            std::visit(
                [&value](auto&& arg) {
                  std::stringstream sstr(value);
                  sstr >> *arg;
                },
                argument.m_value);
          }

          break;
        }
      }

      // Print a warning if there was an unknown argument.
      if(!foundArgument)
      {
        os << "Ignoring unknown command line argument \"" << flag << "\"." << std::endl;
        result = false;
      }

      // Advance to the next flag.
      ++i;

      // If the value was separated, we have to advance our index once more.
      if(foundArgument && valueIsSeparate)
      {
        ++i;
      }
    }

    return result;
  }

private:
  struct Argument
  {
    std::vector<std::string> m_flags;
    Value                    m_value;
    std::string              m_help;
  };

  std::string           m_description;
  std::vector<Argument> m_arguments;
};

//--------------------------------------------------------------------------------------------------
// Command line parser with support for verbs
// Has a regular CommandLineParser, accessible with .global(), to parse and hold global arguments.
// Adds support for sub-commands, e.g.:
//
//   ./exe run --run-flag
//   ./exe { run --run-flag } { run --other-run-flag }
//
// Calling addSubcommand("run") will capture the above --run-flag (and --other-run-flag) in an array
// of SubcommandArgs returned by subcommands(). This can then be parsed by a regular
// CommandLineParser.
//
class MultiCommandLineParser
{
public:
  // Struct to hold arguments for subcommands commands with the syntax {verb
  // ...}. These can then be used like a regular argc count and argv pointer
  // array passed to main().
  struct SubcommandArgs
  {
    std::vector<std::string> args;
    int                      count() const { return static_cast<int>(args.size()); }
    char**                   argv() const
    {
      ptrs.clear();
      for(auto& arg : args)
        ptrs.push_back(const_cast<char*>(arg.c_str()));
      return ptrs.data();
    }

  private:
    mutable std::vector<char*> ptrs;
  };

  MultiCommandLineParser(const std::string& description)
      : m_global(description)
  {
  }

  void printHelp(std::ostream& os = std::cout) const
  {
    os << "Global arguments" << std::endl;
    os << std::endl;
    m_global.printHelp();
    os << std::endl;
    os << "Subcommands" << std::endl;
    os << std::endl;
    for(auto desc : m_descriptions)
    {
      os << "    " << desc.first << ": " << desc.second << std::endl;
    }
    os << std::endl;
    os << "Choose multiple with: '{first --arg} {second --arg}'" << std::endl;
  }

  void addSubcommand(const std::string& verb, const std::string& description) { m_descriptions[verb] = description; }

  bool parse(int argc, char* argv[], std::ostream& os = std::cerr)
  {
    std::string    parsedLine;
    SubcommandArgs globalArgs;

    // Must have at least the executable name
    if(argc < 1)
      return false;

    // All args must start with the executable name
    globalArgs.args.push_back(argv[0]);

    // Skip the first argument (name of the program).
    int  depth             = 0;
    bool needClosingBrace  = false;
    auto currentSubcommand = m_subcommands.end();
    for(int i = 1; i < argc; ++i)
    {
      std::string arg(argv[i]);

      // Check for opening brace.
      if(string_starts_with(arg, '{'))
      {
        if(depth != 0)
        {
          os << "Missing subcommand terminator '}':" << std::endl;
          os << parsedLine << " <- ?" << std::endl;
          return false;
        }
        ++depth;
        needClosingBrace = true;
        arg              = arg.substr(1);
        if(arg.empty())
        {
          continue;
        }
      }

      parsedLine += (parsedLine.empty() ? "" : " ") + arg;

      // Check for closing brace. Can appear after some argument text, so
      // endsSubcommand is used to delay processing.
      bool endsSubcommand = false;
      if(string_ends_with(arg, '}'))
      {
        if(depth != 1 || !needClosingBrace)
        {
          os << "Unexpected '}':" << std::endl;
          os << parsedLine << " <- ?" << std::endl;
          return false;
        }
        arg = arg.substr(0, arg.size() - 1);
        if(arg.empty())
        {
          currentSubcommand = m_subcommands.end();
          --depth;
          continue;
        }
        else
        {
          endsSubcommand = true;
        }
      }

      // Parse subcommand arguments
      if(currentSubcommand != m_subcommands.end())
      {
        currentSubcommand->second.args.push_back(arg);
      }
      else
      {
        // Check for the subcommand verb
        if(m_descriptions.find(arg) == m_descriptions.end())
        {
          if(depth == 1)
          {
            // Inside {..} and the first thing isn't a verb.
            os << "Missing verb for subcommand:" << std::endl;
            os << parsedLine << " <- ?" << std::endl;
            return false;
          }
          else
          {
            // In global scope and not a verb
            globalArgs.args.push_back(arg);
          }
        }
        else
        {
          // Got a verb. Start a subcommand
          if(depth == 0)
          {
            // Allow a single subcommand without braces, but not multiple
            if(!m_subcommands.empty())
            {
              os << "Braces are required for multiple subcommands:" << std::endl;
              os << parsedLine << " <- ?" << std::endl;
              return false;
            }
            ++depth;
          }

          // Create a new subcommand object
          currentSubcommand = m_subcommands.insert(m_subcommands.end(), {arg, {}});

          // Add the executable path as the first argument
          currentSubcommand->second.args.push_back(argv[0]);
        }
      }

      if(endsSubcommand)
      {
        currentSubcommand = m_subcommands.end();
        --depth;
      }
    }

    // Error if missing a '}'
    if(depth == 1 && needClosingBrace)
    {
      os << "Missing '}'" << std::endl;
      return false;
    }

    // Parse the global commands
    return m_global.parse(globalArgs.count(), globalArgs.argv());
  }

  const std::vector<std::pair<std::string, SubcommandArgs>>& subcommands() const { return m_subcommands; }

  const CommandLineParser& global() const { return m_global; }
  CommandLineParser&       global() { return m_global; }

private:
  // Single-character C++17-compatible version of std::starts_with.
  bool string_starts_with(const std::string& str, char c) const { return (str.size() > 0) && (str[0] == c); }
  // Single-character C++17-compatible version of std::ends_with.
  bool string_ends_with(const std::string& str, char c) const { return (str.size() > 0) && (str[str.size() - 1] == c); }

  CommandLineParser                                   m_global;
  std::map<std::string, std::string>                  m_descriptions;
  std::vector<std::pair<std::string, SubcommandArgs>> m_subcommands;
};
