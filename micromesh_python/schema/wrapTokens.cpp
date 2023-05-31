//
// Copyright 2016 Pixar
//
// Licensed under the Apache License, Version 2.0 (the "Apache License")
// with the following modification; you may not use this file except in
// compliance with the Apache License and the following modification to it:
// Section 6. Trademarks. is deleted and replaced with:
//
// 6. Trademarks. This License does not grant permission to use the trade
//    names, trademarks, service marks, or product names of the Licensor
//    and its affiliates, except as required to comply with Section 4(c) of
//    the License and to reproduce the content of the NOTICE file.
//
// You may obtain a copy of the Apache License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the Apache License with the above modification is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the Apache License for the specific
// language governing permissions and limitations under the Apache License.
//
// GENERATED FILE.  DO NOT EDIT.
#include <boost/python/class.hpp>
#include "./tokens.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

// Helper to return a static token as a string.  We wrap tokens as Python
// strings and for some reason simply wrapping the token using def_readonly
// bypasses to-Python conversion, leading to the error that there's no
// Python type for the C++ TfToken type.  So we wrap this functor instead.
class _WrapStaticToken {
public:
    _WrapStaticToken(const TfToken* token) : _token(token) { }

    std::string operator()() const
    {
        return _token->GetString();
    }

private:
    const TfToken* _token;
};

template <typename T>
void
_AddToken(T& cls, const char* name, const TfToken& token)
{
    cls.add_static_property(name,
                            boost::python::make_function(
                                _WrapStaticToken(&token),
                                boost::python::return_value_policy<
                                    boost::python::return_by_value>(),
                                boost::mpl::vector1<std::string>()));
}

} // anonymous

void wrapNvMicromeshTokens()
{
    boost::python::class_<NvMicromeshTokensType, boost::noncopyable>
        cls("Tokens", boost::python::no_init);
    _AddToken(cls, "primvarsMicromeshCompressed", NvMicromeshTokens->primvarsMicromeshCompressed);
    _AddToken(cls, "primvarsMicromeshDirectionBounds", NvMicromeshTokens->primvarsMicromeshDirectionBounds);
    _AddToken(cls, "primvarsMicromeshDirections", NvMicromeshTokens->primvarsMicromeshDirections);
    _AddToken(cls, "primvarsMicromeshFloatBias", NvMicromeshTokens->primvarsMicromeshFloatBias);
    _AddToken(cls, "primvarsMicromeshFloatScale", NvMicromeshTokens->primvarsMicromeshFloatScale);
    _AddToken(cls, "primvarsMicromeshHistogramBlockFormats", NvMicromeshTokens->primvarsMicromeshHistogramBlockFormats);
    _AddToken(cls, "primvarsMicromeshHistogramCounts", NvMicromeshTokens->primvarsMicromeshHistogramCounts);
    _AddToken(cls, "primvarsMicromeshHistogramSubdivLevels", NvMicromeshTokens->primvarsMicromeshHistogramSubdivLevels);
    _AddToken(cls, "primvarsMicromeshMaxSubdivLevel", NvMicromeshTokens->primvarsMicromeshMaxSubdivLevel);
    _AddToken(cls, "primvarsMicromeshMinSubdivLevel", NvMicromeshTokens->primvarsMicromeshMinSubdivLevel);
    _AddToken(cls, "primvarsMicromeshOffsetMap", NvMicromeshTokens->primvarsMicromeshOffsetMap);
    _AddToken(cls, "primvarsMicromeshQuaternionMap", NvMicromeshTokens->primvarsMicromeshQuaternionMap);
    _AddToken(cls, "primvarsMicromeshTriangleBlockFormats", NvMicromeshTokens->primvarsMicromeshTriangleBlockFormats);
    _AddToken(cls, "primvarsMicromeshTriangleFlags", NvMicromeshTokens->primvarsMicromeshTriangleFlags);
    _AddToken(cls, "primvarsMicromeshTriangleFlagsByteSize", NvMicromeshTokens->primvarsMicromeshTriangleFlagsByteSize);
    _AddToken(cls, "primvarsMicromeshTriangleFlagsCount", NvMicromeshTokens->primvarsMicromeshTriangleFlagsCount);
    _AddToken(cls, "primvarsMicromeshTriangleFlagsFormat", NvMicromeshTokens->primvarsMicromeshTriangleFlagsFormat);
    _AddToken(cls, "primvarsMicromeshTriangleMappings", NvMicromeshTokens->primvarsMicromeshTriangleMappings);
    _AddToken(cls, "primvarsMicromeshTriangleMappingsByteSize", NvMicromeshTokens->primvarsMicromeshTriangleMappingsByteSize);
    _AddToken(cls, "primvarsMicromeshTriangleMappingsCount", NvMicromeshTokens->primvarsMicromeshTriangleMappingsCount);
    _AddToken(cls, "primvarsMicromeshTriangleMappingsFormat", NvMicromeshTokens->primvarsMicromeshTriangleMappingsFormat);
    _AddToken(cls, "primvarsMicromeshTriangleMinMaxs", NvMicromeshTokens->primvarsMicromeshTriangleMinMaxs);
    _AddToken(cls, "primvarsMicromeshTriangleMinMaxsByteSize", NvMicromeshTokens->primvarsMicromeshTriangleMinMaxsByteSize);
    _AddToken(cls, "primvarsMicromeshTriangleMinMaxsCount", NvMicromeshTokens->primvarsMicromeshTriangleMinMaxsCount);
    _AddToken(cls, "primvarsMicromeshTriangleMinMaxsFormat", NvMicromeshTokens->primvarsMicromeshTriangleMinMaxsFormat);
    _AddToken(cls, "primvarsMicromeshTriangleSubdivLevels", NvMicromeshTokens->primvarsMicromeshTriangleSubdivLevels);
    _AddToken(cls, "primvarsMicromeshTriangleValueOffsets", NvMicromeshTokens->primvarsMicromeshTriangleValueOffsets);
    _AddToken(cls, "primvarsMicromeshValueByteSize", NvMicromeshTokens->primvarsMicromeshValueByteSize);
    _AddToken(cls, "primvarsMicromeshValueCount", NvMicromeshTokens->primvarsMicromeshValueCount);
    _AddToken(cls, "primvarsMicromeshValueFormat", NvMicromeshTokens->primvarsMicromeshValueFormat);
    _AddToken(cls, "primvarsMicromeshValueFrequency", NvMicromeshTokens->primvarsMicromeshValueFrequency);
    _AddToken(cls, "primvarsMicromeshValueLayout", NvMicromeshTokens->primvarsMicromeshValueLayout);
    _AddToken(cls, "primvarsMicromeshValues", NvMicromeshTokens->primvarsMicromeshValues);
    _AddToken(cls, "primvarsMicromeshVersion", NvMicromeshTokens->primvarsMicromeshVersion);
}
