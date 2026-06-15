/*
    This file is part of darktable,
    Copyright (C) 2010-2026 darktable developers.

    darktable is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    darktable is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with darktable.  If not, see <http://www.gnu.org/licenses/>.
*/

// xtrans_interpolate adapted from dcraw 9.20

#define TS DT_MARKESTEIJN_TS

#define PAD_G1_G3 3
#define PAD_G_INTERP 3
#define PAD_G_RECALC 6

/** Lookup for allhex[], making sure that row/col aren't negative **/
static inline const short *_hexmap(const int row,
                                   const int col,
                                   short (*const allhex)[3][8])
{
  // Row and column offsets may be negative, but C's modulo function
  // is not useful here with a negative dividend. To be safe, add a
  // fairly large multiple of 3. In current code row and col will
  // never be less than -9 (1-pass) or -14 (3-pass).
  int irow = row + 600;
  int icol = col + 600;
  assert(irow >= 0 && icol >= 0);
  return allhex[irow % 3][icol % 3];
}

/*
   Frank Markesteijn's algorithm for Fuji X-Trans sensors
*/
static void xtrans_markesteijn_interpolate(float *out,
                                           const float *const in,
                                           const int width,
                                           const int height,
                                           const uint8_t (*const xtrans)[6],
                                           const int passes,
                                           const uint32_t filters)
{
  // Markesteijn X-Trans interpolation -- Rust FFI
  darkroom_xtrans_markesteijn(out, in, width, height, (const unsigned char *)xtrans, passes, filters);
  const int pad_tile = (passes == 1) ? 12 : 17;
  _vng_lininterpolate(out, in, width, height, filters, xtrans, pad_tile);
}

#undef TS

#define TS DT_FDC_TS
static void xtrans_fdc_interpolate(float *out,
                                   const float *const in,
                                   const int width,
                                   const int height,
                                   const uint8_t (*const xtrans)[6],
                                   const int iso)
{
  // FDC X-Trans interpolation -- Rust FFI. hybrid_fdc selects hybrid vs pure
  // FDC chroma based on ISO (kept in C for the config lookup).
  float hybrid_fdc[2] = { 1.0f, 0.0f };
  const int xover_iso = dt_conf_get_int("plugins/darkroom/demosaic/fdc_xover_iso");
  if(iso > xover_iso)
  {
    hybrid_fdc[0] = 0.0f;
    hybrid_fdc[1] = 1.0f;
  }
  darkroom_xtrans_fdc(out, in, width, height, (const unsigned char *)xtrans, hybrid_fdc[0], hybrid_fdc[1]);
}

#undef PIX_SWAP
#undef PIX_SORT
#undef CCLIP
#undef TS

#ifdef HAVE_OPENCL
static cl_int process_markesteijn_cl(const dt_iop_module_t *self,
                                     const dt_dev_pixelpipe_iop_t *piece,
                                     cl_mem dev_in,
                                     cl_mem dev_out,
                                     cl_mem dev_xtrans,
                                     const uint8_t(*const xtrans)[6],
                                     const int width,
                                     const int height)
{
  const dt_iop_demosaic_data_t *data = piece->data;
  const dt_iop_demosaic_global_data_t *gd = self->global_data;

  const int devid = piece->pipe->devid;

  cl_mem dev_tmptmp = NULL;
  cl_mem dev_rgbv[8] =    { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
  cl_mem dev_drv[8] =     { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
  cl_mem dev_homo[8] =    { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
  cl_mem dev_homosum[8] = { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL };
  cl_mem dev_gminmax = NULL;
  cl_mem dev_allhex = NULL;
  cl_mem dev_aux = NULL;
  cl_int err = CL_MEM_OBJECT_ALLOCATION_FAILURE;

  cl_mem *dev_rgb = dev_rgbv;

  const int passes = ((data->demosaicing_method & ~DT_DEMOSAIC_DUAL) == DT_IOP_DEMOSAIC_MARKESTEIJN_3) ? 3 : 1;
  const int ndir = passes > 1 ? 8 : 4 ;
  const int pad_tile = (passes == 1) ? 12 : 17;
  process_vng_cl(self, piece, dev_in, dev_out, dev_xtrans, xtrans, width, height, 9u, pad_tile+2, TRUE);

  static const short orth[12] = { 1, 0, 0, 1, -1, 0, 0, -1, 1, 0, 0, 1 },
                     patt[2][16] = { { 0, 1, 0, -1, 2, 0, -1, 0, 1, 1, 1, -1, 0, 0, 0, 0 },
                                     { 0, 1, 0, -2, 1, 0, -2, 0, 1, 1, -2, -2, 1, -1, -1, 1 } };

    // allhex contains the offset coordinates (x,y) of a green hexagon around each
    // non-green pixel and vice versa
    char allhex[3][3][8][2];
    // sgreen is the offset in the sensor matrix of the solitary
    // green pixels (initialized here only to avoid compiler warning)
    char sgreen[2] = { 0 };

    // Map a green hexagon around each non-green pixel and vice versa:
    for(int row = 0; row < 3; row++)
      for(int col = 0; col < 3; col++)
        for(int ng = 0, d = 0; d < 10; d += 2)
        {
          const int g = FCNxtrans(row, col, xtrans) == 1;
          if(FCNxtrans(row + orth[d] + 6, col + orth[d + 2] + 6, xtrans) == 1)
            ng = 0;
          else
            ng++;
          // if there are four non-green pixels adjacent in cardinal
          // directions, this is the solitary green pixel
          if(ng == 4)
          {
            sgreen[0] = col;
            sgreen[1] = row;
          }
          if(ng == g + 1)
            for(int c = 0; c < 8; c++)
            {
              const int v = orth[d] * patt[g][c * 2] + orth[d + 1] * patt[g][c * 2 + 1];
              const int h = orth[d + 2] * patt[g][c * 2] + orth[d + 3] * patt[g][c * 2 + 1];

              allhex[row][col][c ^ (g * 2 & d)][0] = h;
              allhex[row][col][c ^ (g * 2 & d)][1] = v;
            }
        }

    dev_allhex = dt_opencl_copy_host_to_device_constant(devid, sizeof(allhex), allhex);
    if(dev_allhex == NULL) goto error;

    for(int n = 0; n < ndir; n++)
    {
      dev_rgbv[n] = dt_opencl_alloc_device_buffer(devid, sizeof(float) * 4 * width * height);
      if(dev_rgbv[n] == NULL) goto error;
    }

    dev_gminmax = dt_opencl_alloc_device_buffer(devid, sizeof(float) * 2 * width * height);
    if(dev_gminmax == NULL) goto error;

    dev_aux = dt_opencl_alloc_device_buffer(devid, sizeof(float) * 4 * width * height);
    if(dev_aux == NULL) goto error;

    // copy from dev_in to first rgb image buffer.
    err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_markesteijn_initial_copy, width, height,
        CLARG(dev_in), CLARG(dev_rgb[0]), CLARG(width), CLARG(height), CLARG(dev_xtrans));
    if(err != CL_SUCCESS) goto error;

    // duplicate dev_rgb[0] to dev_rgb[1], dev_rgb[2], and dev_rgb[3]
    for(int c = 1; c <= 3; c++)
    {
      err = dt_opencl_enqueue_copy_buffer_to_buffer(devid, dev_rgb[0], dev_rgb[c], 0, 0,
                                                    sizeof(float) * 4 * width * height);
      if(err != CL_SUCCESS) goto error;
    }

    // find minimum and maximum allowed green values of red/blue pixel pairs
    dt_opencl_local_buffer_t locopt_g1_g3
      = (dt_opencl_local_buffer_t){ .xoffset = 2*3, .xfactor = 1, .yoffset = 2*3, .yfactor = 1,
                                    .cellsize = 1 * sizeof(float), .overhead = 0,
                                    .sizex = 1 << 8, .sizey = 1 << 8 };

    err = dt_opencl_local_buffer_opt(devid, gd->kernel_markesteijn_green_minmax, &locopt_g1_g3);
    if(err != CL_SUCCESS) goto error;

    {
      const size_t sizes[2] = { ROUNDUP(width, locopt_g1_g3.sizex), ROUNDUP(height, locopt_g1_g3.sizey) };
      const size_t local[2] = { locopt_g1_g3.sizex, locopt_g1_g3.sizey };
      err = dt_opencl_enqueue_kernel_2d_local_args(devid, gd->kernel_markesteijn_green_minmax, sizes, local,
        CLARG(dev_rgb[0]), CLARG(dev_gminmax),
        CLARG(width), CLARG(height), CLARGINT(PAD_G1_G3), CLARRAY(2, sgreen),
        CLARG(dev_xtrans), CLARG(dev_allhex), CLLOCAL(sizeof(float) * (locopt_g1_g3.sizex + 2*3) * (locopt_g1_g3.sizey + 2*3)));
      if(err != CL_SUCCESS) goto error;
    }

    // interpolate green horizontally, vertically, and along both diagonals
    dt_opencl_local_buffer_t locopt_g_interp
      = (dt_opencl_local_buffer_t){ .xoffset = 2*6, .xfactor = 1, .yoffset = 2*6, .yfactor = 1,
                                    .cellsize = 4 * sizeof(float), .overhead = 0,
                                    .sizex = 1 << 8, .sizey = 1 << 8 };

    err = dt_opencl_local_buffer_opt(devid, gd->kernel_markesteijn_interpolate_green, &locopt_g_interp);
    if(err != CL_SUCCESS) goto error;

    {
      const size_t sizes[2] = { ROUNDUP(width, locopt_g_interp.sizex), ROUNDUP(height, locopt_g_interp.sizey) };
      const size_t local[2] = { locopt_g_interp.sizex, locopt_g_interp.sizey };
      err = dt_opencl_enqueue_kernel_2d_local_args(devid, gd->kernel_markesteijn_interpolate_green, sizes, local,
        CLARG(dev_rgb[0]), CLARG(dev_rgb[1]), CLARG(dev_rgb[2]), CLARG(dev_rgb[3]),
        CLARG(dev_gminmax), CLARG(width), CLARG(height),
        CLARGINT(PAD_G_INTERP), CLARRAY(2, sgreen), CLARG(dev_xtrans),
        CLARG(dev_allhex), CLLOCAL(sizeof(float) * 4 * (locopt_g_interp.sizex + 2*6) * (locopt_g_interp.sizey + 2*6)));
      if(err != CL_SUCCESS) goto error;
    }

    // multi-pass loop: one pass for Markesteijn-1 and three passes for Markesteijn-3
    for(int pass = 0; pass < passes; pass++)
    {

      // if on second pass, copy rgb[0] to [3] into rgb[4] to [7] ....
      if(pass == 1)
      {
        for(int c = 0; c < 4; c++)
        {
          err = dt_opencl_enqueue_copy_buffer_to_buffer(devid, dev_rgb[c], dev_rgb[c + 4], 0, 0,
                                                        sizeof(float) * 4 * width * height);
          if(err != CL_SUCCESS) goto error;
        }
        // ... and process that second set of buffers
        dev_rgb += 4;
      }

      // second and third pass (only Markesteijn-3)
      if(pass)
      {
        // recalculate green from interpolated values of closer pixels
        err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_markesteijn_recalculate_green, width, height,
          CLARG(dev_rgb[0]), CLARG(dev_rgb[1]), CLARG(dev_rgb[2]), CLARG(dev_rgb[3]), CLARG(dev_gminmax),
          CLARG(width), CLARG(height), CLARGINT(PAD_G_RECALC), CLARRAY(2, sgreen),
          CLARG(dev_xtrans), CLARG(dev_allhex));
        if(err != CL_SUCCESS) goto error;
      }

      // interpolate red and blue values for solitary green pixels
      const int pad_rb_g = (passes == 1) ? 6 : 5;
      dt_opencl_local_buffer_t locopt_rb_g
        = (dt_opencl_local_buffer_t){ .xoffset = 2*2, .xfactor = 1, .yoffset = 2*2, .yfactor = 1,
                                      .cellsize = 4 * sizeof(float), .overhead = 0,
                                      .sizex = 1 << 8, .sizey = 1 << 8 };

      err = dt_opencl_local_buffer_opt(devid, gd->kernel_markesteijn_solitary_green, &locopt_rb_g);
      if(err != CL_SUCCESS) goto error;

      cl_mem *dev_trgb = dev_rgb;
      for(int d = 0, i = 1, h = 0; d < 6; d++, i ^= 1, h ^= 2)
      {
        const char dir[2] = { i, i ^ 1 };

        // we use dev_aux to transport intermediate results from one loop run to the next
        const size_t sizes[2] = { ROUNDUP(width, locopt_rb_g.sizex), ROUNDUP(height, locopt_rb_g.sizey) };
        const size_t local[2] = { locopt_rb_g.sizex, locopt_rb_g.sizey };
        err = dt_opencl_enqueue_kernel_2d_local_args(devid, gd->kernel_markesteijn_solitary_green, sizes, local,
          CLARG(dev_trgb[0]), CLARG(dev_aux), CLARG(width), CLARG(height), CLARG(pad_rb_g),
          CLARG(d), CLARRAY(2, dir), CLARG(h), CLARRAY(2, sgreen), CLARG(dev_xtrans), CLLOCAL(sizeof(float) * 4 * (locopt_rb_g.sizex + 2*2) * (locopt_rb_g.sizey + 2*2)));
        if(err != CL_SUCCESS) goto error;

        if((d < 2) || (d & 1)) dev_trgb++;
      }

      // interpolate red for blue pixels and vice versa
      const int pad_rb_br = (passes == 1) ? 6 : 5;
      dt_opencl_local_buffer_t locopt_rb_br
        = (dt_opencl_local_buffer_t){ .xoffset = 2*3, .xfactor = 1, .yoffset = 2*3, .yfactor = 1,
                                      .cellsize = 4 * sizeof(float), .overhead = 0,
                                      .sizex = 1 << 8, .sizey = 1 << 8 };

      err = dt_opencl_local_buffer_opt(devid, gd->kernel_markesteijn_red_and_blue, &locopt_rb_br);
      if(err != CL_SUCCESS) goto error;

      for(int d = 0; d < 4; d++)
      {
        const size_t sizes[2] = { ROUNDUP(width, locopt_rb_br.sizex), ROUNDUP(height, locopt_rb_br.sizey) };
        const size_t local[2] = { locopt_rb_br.sizex, locopt_rb_br.sizey };
        err = dt_opencl_enqueue_kernel_2d_local_args(devid, gd->kernel_markesteijn_red_and_blue, sizes, local,
          CLARG(dev_rgb[d]), CLARG(width), CLARG(height), CLARG(pad_rb_br), CLARG(d), CLARRAY(2, sgreen),
          CLARG(dev_xtrans), CLLOCAL(sizeof(float) * 4 * (locopt_rb_br.sizex + 2*3) * (locopt_rb_br.sizey + 2*3)));
        if(err != CL_SUCCESS) goto error;
      }

      // interpolate red and blue for 2x2 blocks of green
      const int pad_g22 = (passes == 1) ? 8 : 4;
      dt_opencl_local_buffer_t locopt_g22
        = (dt_opencl_local_buffer_t){ .xoffset = 2*2, .xfactor = 1, .yoffset = 2*2, .yfactor = 1,
                                      .cellsize = 4 * sizeof(float), .overhead = 0,
                                      .sizex = 1 << 8, .sizey = 1 << 8 };

      err = dt_opencl_local_buffer_opt(devid, gd->kernel_markesteijn_interpolate_twoxtwo, &locopt_g22);
      if(err != CL_SUCCESS) goto error;

      for(int d = 0, n = 0; d < ndir; d += 2, n++)
      {
        const size_t sizes[2] = { ROUNDUP(width, locopt_g22.sizex), ROUNDUP(height, locopt_g22.sizey) };
        const size_t local[2] = { locopt_g22.sizex, locopt_g22.sizey };
        err = dt_opencl_enqueue_kernel_2d_local_args(devid, gd->kernel_markesteijn_interpolate_twoxtwo, sizes, local,
          CLARG(dev_rgb[n]), CLARG(width), CLARG(height), CLARG(pad_g22), CLARG(d), CLARRAY(2, sgreen),
          CLARG(dev_xtrans), CLARG(dev_allhex), CLLOCAL(sizeof(float) * 4 * (locopt_g22.sizex + 2*2) * (locopt_g22.sizey + 2*2)));
        if(err != CL_SUCCESS) goto error;
      }
    }
    // end of multi pass

    // gminmax data no longer needed
    dt_opencl_release_mem_object(dev_gminmax);
    dev_gminmax = NULL;

    // jump back to the first set of rgb buffers (this is a noop for Markesteijn-1)
    dev_rgb = dev_rgbv;

    err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    // prepare derivatives buffers
    for(int n = 0; n < ndir; n++)
    {
      dev_drv[n] = dt_opencl_alloc_device_buffer(devid, sizeof(float) * width * height);
      if(dev_drv[n] == NULL) goto error;
    }

    // convert to perceptual colorspace and differentiate in all directions
    const int pad_yuv = (passes == 1) ? 8 : 13;
    dt_opencl_local_buffer_t locopt_diff
      = (dt_opencl_local_buffer_t){ .xoffset = 2*1, .xfactor = 1, .yoffset = 2*1, .yfactor = 1,
                                    .cellsize = 4 * sizeof(float), .overhead = 0,
                                    .sizex = 1 << 8, .sizey = 1 << 8 };

    err = dt_opencl_local_buffer_opt(devid, gd->kernel_markesteijn_differentiate, &locopt_diff);
    if(err != CL_SUCCESS) goto error;

    for(int d = 0; d < ndir; d++)
    {
      // convert to perceptual YPbPr colorspace
      err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_markesteijn_convert_yuv, width, height,
        CLARG(dev_rgb[d]), CLARG(dev_aux), CLARG(width), CLARG(height), CLARG(pad_yuv));
      if(err != CL_SUCCESS) goto error;


      // differentiate in all directions
      const size_t sizes_diff[2] = { ROUNDUP(width, locopt_diff.sizex), ROUNDUP(height, locopt_diff.sizey) };
      const size_t local_diff[2] = { locopt_diff.sizex, locopt_diff.sizey };
      err = dt_opencl_enqueue_kernel_2d_local_args(devid, gd->kernel_markesteijn_differentiate, sizes_diff, local_diff,
        CLARG(dev_aux), CLARG(dev_drv[d]),
        CLARG(width), CLARG(height), CLARG(pad_yuv), CLARG(d), CLLOCAL(sizeof(float) * 4 * (locopt_diff.sizex + 2*1) * (locopt_diff.sizey + 2*1)));
      if(err != CL_SUCCESS) goto error;
    }

    // reserve buffers for homogeneity maps and sum maps
    err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    for(int n = 0; n < ndir; n++)
    {
      dev_homo[n] = dt_opencl_alloc_device_buffer(devid, sizeof(unsigned char) * width * height);
      if(dev_homo[n] == NULL) goto error;

      dev_homosum[n] = dt_opencl_alloc_device_buffer(devid, sizeof(unsigned char) * width * height);
      if(dev_homosum[n] == NULL) goto error;
    }

    // get thresholds for homogeneity map (store them in dev_aux)
    for(int d = 0; d < ndir; d++)
    {
      const int pad_homo = (passes == 1) ? 10 : 15;
      err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_markesteijn_homo_threshold, width, height,
        CLARG(dev_drv[d]), CLARG(dev_aux), CLARG(width), CLARG(height), CLARG(pad_homo), CLARG(d));
      if(err != CL_SUCCESS) goto error;
    }

    // set homogeneity maps
    const int pad_homo = (passes == 1) ? 10 : 15;
    dt_opencl_local_buffer_t locopt_homo
      = (dt_opencl_local_buffer_t){ .xoffset = 2*1, .xfactor = 1, .yoffset = 2*1, .yfactor = 1,
                                    .cellsize = 1 * sizeof(float), .overhead = 0,
                                    .sizex = 1 << 8, .sizey = 1 << 8 };

    err = dt_opencl_local_buffer_opt(devid, gd->kernel_markesteijn_homo_set, &locopt_homo);
    if(err != CL_SUCCESS) goto error;

    for(int d = 0; d < ndir; d++)
    {
      const size_t sizes[2] = { ROUNDUP(width, locopt_homo.sizex),ROUNDUP(height, locopt_homo.sizey) };
      const size_t local[2] = { locopt_homo.sizex, locopt_homo.sizey };
      err = dt_opencl_enqueue_kernel_2d_local_args(devid, gd->kernel_markesteijn_homo_set, sizes, local,
        CLARG(dev_drv[d]), CLARG(dev_aux),
        CLARG(dev_homo[d]), CLARG(width), CLARG(height), CLARG(pad_homo), CLLOCAL(sizeof(float) * (locopt_homo.sizex + 2*1) * (locopt_homo.sizey + 2*1)));
      if(err != CL_SUCCESS) goto error;
    }

    // get rid of dev_drv buffers
    for(int n = 0; n < 8; n++)
    {
      dt_opencl_release_mem_object(dev_drv[n]);
      dev_drv[n] = NULL;
    }

    // build 5x5 sum of homogeneity maps for each pixel and direction
    dt_opencl_local_buffer_t locopt_homo_sum
      = (dt_opencl_local_buffer_t){ .xoffset = 2*2, .xfactor = 1, .yoffset = 2*2, .yfactor = 1,
                                    .cellsize = 1 * sizeof(float), .overhead = 0,
                                    .sizex = 1 << 8, .sizey = 1 << 8 };

    err = dt_opencl_local_buffer_opt(devid, gd->kernel_markesteijn_homo_sum, &locopt_homo_sum);
    if(err != CL_SUCCESS) goto error;

    for(int d = 0; d < ndir; d++)
    {
      size_t sizes[2] = { ROUNDUP(width, locopt_homo_sum.sizex), ROUNDUP(height, locopt_homo_sum.sizey) };
      size_t local[2] = { locopt_homo_sum.sizex, locopt_homo_sum.sizey };
      err = dt_opencl_enqueue_kernel_2d_local_args(devid, gd->kernel_markesteijn_homo_sum, sizes, local,
        CLARG(dev_homo[d]), CLARG(dev_homosum[d]),
        CLARG(width), CLARG(height), CLARG(pad_tile), CLLOCAL(sizeof(char) * (locopt_homo_sum.sizex + 2*2) * (locopt_homo_sum.sizey + 2*2)));
      if(err != CL_SUCCESS) goto error;
    }

    // get maximum of homogeneity maps (store in dev_aux)
    for(int d = 0; d < ndir; d++)
    {
      err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_markesteijn_homo_max, width, height,
        CLARG(dev_homosum[d]), CLARG(dev_aux), CLARG(width), CLARG(height), CLARG(pad_tile), CLARG(d));
      if(err != CL_SUCCESS) goto error;
    }

    // adjust maximum value
    err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_markesteijn_homo_max_corr, width, height,
        CLARG(dev_aux), CLARG(width), CLARG(height), CLARG(pad_tile));
    if(err != CL_SUCCESS) goto error;

    // for Markesteijn-3: use only one of two directions if there is a difference in homogeneity
    for(int d = 0; d < ndir - 4; d++)
    {
      err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_markesteijn_homo_quench, width, height,
        CLARG(dev_homosum[d]), CLARG(dev_homosum[d + 4]), CLARG(width), CLARG(height), CLARG(pad_tile));
      if(err != CL_SUCCESS) goto error;
    }

    // initialize output buffer to zero
    err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_markesteijn_zero, width, height,
        CLARG(dev_out), CLARG(width), CLARG(height), CLARG(pad_tile));
    if(err != CL_SUCCESS) goto error;

    // need to get another temp image for the output image
    err = CL_MEM_OBJECT_ALLOCATION_FAILURE;
    dev_tmptmp = dt_opencl_alloc_device(devid, width, height, sizeof(float) * 4);
    if(dev_tmptmp == NULL) goto error;

    cl_mem dev_t1 = dev_out;
    cl_mem dev_t2 = dev_tmptmp;

    // accumulate all contributions
    for(int d = 0; d < ndir; d++)
    {
      err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_markesteijn_accu, width, height,
        CLARG(dev_t1), CLARG(dev_t2), CLARG(dev_rgbv[d]), CLARG(dev_homosum[d]), CLARG(dev_aux), CLARG(width),
        CLARG(height), CLARG(pad_tile));
      if(err != CL_SUCCESS) goto error;

      // swap buffers
      cl_mem dev_t = dev_t2;
      dev_t2 = dev_t1;
      dev_t1 = dev_t;
    }

    // copy output to dev_tmptmp (if not already there)
    // note: we need to take swap of buffers into account, so current output lies in dev_t1
    if(dev_t1 != dev_tmptmp)
    {
      const size_t region[2] = { width, height };
      err = dt_opencl_enqueue_copy_image(devid, dev_t1, dev_tmptmp, CLIMG_ORIGIN, CLIMG_ORIGIN, region);
      if(err != CL_SUCCESS) goto error;
    }

    // process the final image
    err = dt_opencl_enqueue_kernel_2d_args(devid, gd->kernel_markesteijn_final, width, height,
        CLARG(dev_tmptmp), CLARG(dev_out), CLARG(width), CLARG(height), CLARG(pad_tile));

error:
  for(int n = 0; n < 8; n++)
    dt_opencl_release_mem_object(dev_rgbv[n]);
  for(int n = 0; n < 8; n++)
    dt_opencl_release_mem_object(dev_drv[n]);
  for(int n = 0; n < 8; n++)
    dt_opencl_release_mem_object(dev_homo[n]);
  for(int n = 0; n < 8; n++)
    dt_opencl_release_mem_object(dev_homosum[n]);
  dt_opencl_release_mem_object(dev_gminmax);
  dt_opencl_release_mem_object(dev_tmptmp);
  dt_opencl_release_mem_object(dev_allhex);
  dt_opencl_release_mem_object(dev_aux);
  if(err != CL_SUCCESS)
    dt_print(DT_DEBUG_OPENCL, "[opencl_demosaic] markesteijn%s problem '%s'",
      passes == 3 ? "_3" : "", cl_errstr(err));
  return err;
}

#endif

// clang-format off
// modelines: These editor modelines have been set for all relevant files by tools/update_modelines.py
// vim: shiftwidth=2 expandtab tabstop=2 cindent
// kate: tab-indents: off; indent-width 2; replace-tabs on; indent-mode cstyle; remove-trailing-spaces modified;
// clang-format on

