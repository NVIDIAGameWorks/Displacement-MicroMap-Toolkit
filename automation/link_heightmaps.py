#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) <year> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import pathlib
import re
import json
import os
import sys
import urllib
from difflib import SequenceMatcher

parser = argparse.ArgumentParser(description='Searches for and links displacement/heightmap files into gltf materials as a KHR_materials_displacement extension')
parser.add_argument('gltf', nargs='+', type=pathlib.Path, help='glTF filenames')
parser.add_argument('--force', action='store_true', help='Write files without asking first')
parser.add_argument('--quiet', action='store_true', help='Do not print details')
parser.add_argument('--verbose', action='store_true', help='Print search attempts')
parser.add_argument('--scale', type=float, default=1.0, help='Heightmap scale')
parser.add_argument('--bias', type=float, default=0.0, help='Heightmap bias')
parser.add_argument('--extra-paths', nargs='+', type=pathlib.Path, help='Extra heightmap search paths')
parser.add_argument('--filter', nargs='+', help='Regex for material names to change')
parser.add_argument('--filter-out', nargs='+', help='Regex for material names to ignore')
parser.add_argument('--copy-from', type=pathlib.Path, help='Source gltf file to use as a template. Takes precedence over the image filename search.')
#parser.add_argument('--heightmaps', nargs='+', type=pathlib.Path, help='Provide heightmap names explicitly in the order of materials')
args = parser.parse_args()

# Return true if the filename could be a heightmap
def heightmapishName(name):
    return re.search(r'height|disp', name, re.IGNORECASE)

# Seach objects in the gltf json and return the ID for something referencing the given ID
def find_ids(data, object_name, target_names, target_id):
    for id, obj in enumerate(data.get(object_name, [])):
        for target_name in target_names:
            #if args.verbose: print('Looking for {}=={} in {} {}'.format(target_name, target_id, object_name, id))
            if target_name in obj:
                if obj[target_name] == target_id:
                    yield id
                elif isinstance(obj[target_name], dict) and obj[target_name].get('index') == target_id:
                    yield id

def heightmapMatchMetric(candidate, image, material_names):
    candidate_name = str(candidate.relative_to(filepath.parent))
    # Helps when height or displacement is a common term for all textures
    heightmap_terms = re.findall(r'height|disp', candidate_name, re.IGNORECASE)
    image_name = str(image.relative_to(filepath.parent))
    image_seq = SequenceMatcher(None, candidate_name, image_name)
    image_similarity = image_seq.ratio()
    match_seq = SequenceMatcher(None, candidate_name.lower(), '$'.join(material_names).lower())
    material_similarity = match_seq.ratio()
    score = image_similarity + material_similarity * 0.1 + len(heightmap_terms)
    if args.verbose:
        print('Image similarity score {} (filename {}, material name {}, heightmap terms {}):'.format(score, image_similarity, material_similarity * 0.1, len(heightmap_terms)))
        print('  {}'.format(candidate_name))
        print('  {}'.format(image_name))
        print('  material: {}'.format(material_names))
        print('  heightmap terms: {}'.format(", ".join(heightmap_terms)))
        print()
    return score

def filterMaterialName(material_name):
    if args.filter and not any(re.search(pattern, material_name) for patterrn in args.filter):
        if args.verbose:
            print('Skipping material "{}" due to --filter list'.format(material_name))
        return False
    if args.filter_out and any(re.search(pattern, material_name) for pattern in args.filter_out):
        if args.verbose:
            print('Skipping material "{}" due to --filter-out list'.format(material_name))
        return False
    return True

copy_from = None
if args.copy_from:
    with open(args.copy_from) as gltf:
        copy_from = json.load(gltf)

# For all input gltf files
for filepath in args.gltf:
    with open(filepath) as gltf:
        data = json.load(gltf)
    
    # Get a list of existing images
    images = set()
    image_ids = {}
    for id, image in enumerate(data.get('images', [])):
        uri = image.get('uri')
        if not uri: continue
        path = filepath.parent / urllib.parse.unquote(uri)
        #if args.verbose: print('Existing image {}'.format(path))
        images.add(path)
        image_ids[path] = id

    image_types = set()
    image_dirs = set(args.extra_paths or [])
    for image in images:
        image_types.add(image.suffix)
        image_dirs.add(image.parent)

    # Find materials that reference each image
    image_material_ids = {}
    for image in images:
        # Find textures referencing this image
        image_id = image_ids[image]
        texture_ids = list(find_ids(data, 'textures', ['source'], image_id))
        if not texture_ids:
            if args.verbose: print('No texture using image {}, "{}"'.format(image_id, image.relative_to(filepath.parent)))
            continue

        # Find materials referencing these texture ids
        material_ids = []
        for texture_id in texture_ids:
            sampler_id = data['textures'][texture_id]['sampler']
            texture_names = ['baseColorTexture', 'metallicRoughnessTexture', 'emissiveTexture', 'normalTexture', 'occlusionTexture']
            material_ids += list(find_ids(data, 'materials', texture_names, texture_id))
        if not material_ids:
            if args.verbose: print('No material using textures {} (using image {}, "{}")'.format(texture_ids, image_id, image.relative_to(filepath.parent)))
            continue

        image_material_ids[image] = material_ids

    # Search directories of existing images for heightmaps
    candidates = set()
    for dir in image_dirs:
        if args.verbose: print('Searching "{}"...'.format(str(dir)))
        for file in os.listdir(dir):
            candidate = dir / file
            if image.suffix == candidate.suffix and candidate not in images and heightmapishName(file):
                if args.verbose: print('Found candidate {}'.format(str(candidate)))
                candidates.add(candidate)

    # Compute similarities between heightmap names, image names and image materials
    matches = []
    for candidate in candidates:
        for image in images:
            assert candidate != image  # should have been filtered out already
            if image not in image_material_ids: continue
            material_ids = image_material_ids[image]
            material_names = list(data['materials'][material_id]['name'] for material_id in material_ids)
            matches += [(heightmapMatchMetric(candidate, image, material_names), candidate, image, material_ids)]

    # Assume the most similar heightmap names belong to the same materials as existing images in a greedy fashion
    unlinked_heightmaps = []
    unlinked_material_ids = set()
    for metric, candidate, image, material_ids in reversed(sorted(matches)):
        heightmap = candidate
        for material_id in set(material_ids) - unlinked_material_ids:
            unlinked_material_ids.add(material_id)

            # Filter out materials with existing displacement
            material = data['materials'][material_id]
            if 'extensions' not in material:
                material['extensions'] = {}
            extensions = material['extensions']
            if 'KHR_materials_displacement' in extensions:
                if not args.quiet:
                    existing_texture_id = extensions['KHR_materials_displacement']['displacementGeometryTexture']['index']
                    existing_image_id = data['textures'][existing_texture_id]['source']
                    existing_image = filepath.parent / urllib.parse.unquote(data['images'][existing_image_id].get('uri'))
                    material_name = data['materials'][material_id]['name']
                    if heightmap == existing_image:
                        print('Heightmap "{}" is already assigned to material "{}"'.format(
                            candidate.relative_to(filepath.parent), material_name))
                    else:
                        print('Not adding "{}" to material "{}": KHR_materials_displacement already exists (using image {})'.format(
                            candidate.relative_to(filepath.parent), material_name, existing_image.relative_to(filepath.parent)))
                continue

            # Add the material to the job list
            material_name = data['materials'][material_id]['name']
            unlinked_heightmaps += [(material_id, material_name, heightmap, image, sampler_id)]

    had_changes = False

    if not unlinked_heightmaps:
        if not args.quiet:
            print('No extra heightmaps found for {}'.format(str(filepath)))

    # Write new material, texture and image objects into the gltf
    added_texture_map = {}
    heightmaps_added = []
    for job in unlinked_heightmaps:
        (material_id, material_name, heightmap, image, sampler_id) = job
        material = data['materials'][material_id]
        if not filterMaterialName(material_name):
            continue
        if 'extensions' not in material:
            material['extensions'] = {}
        extensions = material['extensions']
        if 'KHR_materials_displacement' in extensions:
            existing_texture_id = extensions['KHR_materials_displacement']['displacementGeometryTexture']['index']
            existing_image_id = data['textures'][existing_texture_id]['source']
            existing_image_uri = data['images'][existing_image_id].get('uri')
            print('Error adding "{}": KHR_materials_displacement already exists for material {} (using image {}). Skipping'.format(
                heightmap.relative_to(filepath.parent), material_name, existing_image_uri), file=sys.stderr)
            continue
        heightmaps_added += [job]
        had_changes = True
        if heightmap.name not in added_texture_map:
            image_id = len(data['images'])
            data['images'] += [{
                'name': str(heightmap.name),
                'uri': urllib.parse.quote(str(heightmap.relative_to(filepath.parent)))
            }]
            texture_id = len(data['textures'])
            data['textures'] += [{
                'sampler': sampler_id,
                'source': image_id
            }]
            if args.verbose:
                print("Pending image {} texture {}: {}".format(image_id, texture_id, heightmap.name))
            added_texture_map[heightmap.name] = texture_id
        extensions['KHR_materials_displacement'] = {
            "displacementGeometryFactor": args.scale,
            "displacementGeometryOffset": args.bias,
            "displacementGeometryTexture": {
                "index" : added_texture_map[heightmap.name]
            },
        }

    if copy_from:
        for material in data['materials']:
            if not filterMaterialName(material['name']):
                continue
            disp = material.get('extensions', {}).get('KHR_materials_displacement')
            if not disp: continue
            for other_material in copy_from['materials']:
                other_disp = other_material.get('extensions', {}).get('KHR_materials_displacement')
                if other_disp and other_material['name'] == material['name'] and 'extensions' in other_material:
                    scale = other_disp['displacementGeometryFactor']
                    bias = other_disp['displacementGeometryOffset']
                    if disp['displacementGeometryFactor'] != scale or disp['displacementGeometryOffset'] != bias:
                        print('Updating material "{}" with scale {} and bias {} from source gltf'.format(material['name'], scale, bias))
                        disp['displacementGeometryFactor'] = scale
                        disp['displacementGeometryOffset'] = bias
                        had_changes = True

    # Skip this gltf file if no changes were made
    if not had_changes:
        continue

    # Ask for permission
    if not args.force:
        if not args.quiet:
            print('Found the following heightmaps:')
            for material_id, material_name, heightmap, image, sampler_id in heightmaps_added:
                print('  Add "{}" to material "{}" (matching image "{}")'.format(
                    heightmap.relative_to(filepath.parent), material_name, image.name))
        answer = input('Write to {}? [Yes/No/Abort] (default: No): '.format(str(filepath))).lower()
        if answer in {'y', 'yes'}:
            pass
        elif answer in {'a', 'abort'}:
            break
        else:
            continue

    # Save the result
    with open(filepath, 'w') as gltf:
        json.dump(data, gltf, indent=2, sort_keys=True)
