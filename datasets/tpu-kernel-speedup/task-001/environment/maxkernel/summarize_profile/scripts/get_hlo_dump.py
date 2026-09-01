#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extracts HLO proto from xplane.pb files if available."""

import argparse
import gzip
import sys
from typing import Optional

from tensorflow.tsl.profiler.protobuf import xplane_pb2


def get_hlo_dump(xplane_path: str,
                 hlo_module_name: Optional[str] = None) -> str:
    """Extracts HLO proto from xplane.pb if available.

  Args:
      xplane_path: Path to .xplane.pb file.
      hlo_module_name: Optional name filter.

  Returns:
      Status string indicating where HLO was saved or if not found.
  """
    try:
        open_func = gzip.open if xplane_path.endswith(".gz") else open
        with open_func(xplane_path, "rb") as f:
            xspace = xplane_pb2.XSpace()
            xspace.ParseFromString(f.read())

        for plane in xspace.planes:
            for stat in plane.stats:
                # Check known XStat metadata names for HLO
                # This is heuristic-based; often HLO protos are embedded as bytes
                # in stats with specific names like 'hlo_proto' or similar.
                # Since we don't have the exact meta name map handy, we might need to search metadata.
                # Simplification: In XPlane, HLOs are often in a dedicated plane or attached to 'device' plane stats.
                pass
                # For now, returning a placeholder as true extraction requires inspecting specific metadata IDs

        return (
            "HLO extraction not fully implemented in this standalone version yet"
            " (requires metadata ID mapping). Please use `load_xplane_and_query` to"
            " explore 'hlo' related events.")

    except Exception as e:
        return f"Error extracting HLO: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="Extract HLO dump from an XProf xplane.pb file.")
    parser.add_argument("xplane_path", help="Path to the .xplane.pb file.")
    parser.add_argument(
        "--module-name",
        "--module_name",
        dest="module_name",
        default=None,
        help="Optional HLO module name filter.",
    )

    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    args = parser.parse_args(argv)

    result = get_hlo_dump(args.xplane_path, args.module_name)
    print(result)


if __name__ == "__main__":
    main()
