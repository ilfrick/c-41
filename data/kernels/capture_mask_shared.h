/*
    This file is part of darktable,
    Copyright (C) 2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable. If not, see <http://www.gnu.org/licenses/>.

    Shared 21-cell safe-mask stencil for the capture-sharpen OpenCL kernel and
    its host-side regression test. Inputs are binary: zero is clipped/dark,
    non-zero is safe.
*/
#ifndef C41_CAPTURE_MASK_SHARED_H
#define C41_CAPTURE_MASK_SHARED_H

#define C41_CAPTURE_MASK_INTERIOR_21(col, row, width, height) \
  ((col) >= 2 && (row) >= 2 && (col) < (width) - 2 && (row) < (height) - 2)

#define C41_CAPTURE_MASK_SAFE_21(in, width) \
  (((in)[-2 * (width) - 1] != 0.0f) && ((in)[-2 * (width)] != 0.0f) && ((in)[-2 * (width) + 1] != 0.0f) \
   && ((in)[-(width) - 2] != 0.0f) && ((in)[-(width) - 1] != 0.0f) && ((in)[-(width)] != 0.0f) \
   && ((in)[-(width) + 1] != 0.0f) && ((in)[-(width) + 2] != 0.0f) \
   && ((in)[-2] != 0.0f) && ((in)[-1] != 0.0f) && ((in)[0] != 0.0f) \
   && ((in)[1] != 0.0f) && ((in)[2] != 0.0f) \
   && ((in)[(width) - 2] != 0.0f) && ((in)[(width) - 1] != 0.0f) \
   && ((in)[(width)] != 0.0f) && ((in)[(width) + 1] != 0.0f) && ((in)[(width) + 2] != 0.0f) \
   && ((in)[2 * (width) - 1] != 0.0f) && ((in)[2 * (width)] != 0.0f) && ((in)[2 * (width) + 1] != 0.0f))

#endif
