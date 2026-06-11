/*
    This file is part of darktable,
    Copyright (C) 2022-2026 darktable developers.

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

/*
Segmentation based highlight reconstruction Version 2

** Overview **

V2 of the segmentation based highlight reconstruction algorithm works for bayer and xtrans sensors.
It has been developed in collaboration by Iain and garagecoder from the gmic team and Hanno Schwalm from dt.

The original idea was presented by Iain in pixls.us in: https://discuss.pixls.us/t/highlight-recovery-teaser/17670
and has been extensively discussed over the last year, a number of different approaches have been evaluated.

No other external modules (like gmic …) are used, the current code has been tuned for performance using omp,
no OpenCL codepath yet.

** Main ideas **

The algorithm follows these basic ideas:
1. We approximate each of the red, green and blue color channels from sensor data in a 3x3 photosite region.
2. We analyse all data on the channels independently.
3. We want to keep details as much as possible
4. In all 3 color planes we look for isolated areas being clipped (segments).
   These segments also include the unclipped photosites at the borders, we also use these locations for estimating the global chrominance.
   Inside these segments we look for a candidate to represent the value we take for restoration.
   Choosing the candidate is done at all non-clipped locations of a segment, the best candidate is selected via a weighting
   function - the weight is derived from
   - the local standard deviation in a 5x5 area and
   - the median value of unclipped positions also in a 5x5 area.
   The best candidate points to the location in the color plane holding the reference value.
   If there is no good candidate we use an averaging approximation over the whole segment with correction of chrominance.
5. A core principle is to inpaint pseudo-chromacity, calculated by subtracting opponent channel means rather than luminance.
6. Cube root is used instead of logarithm for better stability, which suffices for an estimate.

The chosen segmentation algorithm works like this:
1. Doing the segmentation in every color plane.
2. To combine small segments for a shared candidate we use a morphological closing operation, the radius of that UI op
   can be chosen interactively between 0 and 8.
3. The segmentation algorithm uses a modified floodfill, it also takes care of the surrounding rectangle of every segment
   and marks the segment borders.
4. After segmentation we check every segment for
   - the segment's best candidate via the weighting function
   - the candidates location
*/

/* Rebuild algorithm
  In areas with all planes clipped we try to reconstruct (hopefully a good guess) data based on the border gradients and the
  segment's size - here we use a distance transformation.
  What do we need to do so?
  1. We need a "luminance" plane, we use Y0 for this.
  2. We have an additional mask holding information about all-channels-clipped
  3. Based on the Y0 data and all-clipped info we prepare a gradient plane.
  4. We also do a segmentation for the all-clipped data.

  After this preparation steps we reconstruct data for every segment.
  1. Calculate average gradients in an iterative loop for every distance value.
     The new gradients calculation uses the distance and averaged gradients of the last iterative step.
     By doing so we avoid direction problems.
  2. Do a box-blur to suppress ridges, the radius depends on segment size.
  3. Possibly add some noise.
  4. Do a sigmoid correction suppressing artefacts at the borders.
     and write back data from this segment to the gradients plane

  The UI offers
  1. A drop down menu defining the recovery mode
  2. strength slider - this also has a mask button
  3. noise slider
*/

#define HL_RGB_PLANES 3
#define HL_SEGMENT_PLANES 4
#define HL_FLOAT_PLANES 8
#define HL_BORDER 8

static inline float _local_std_deviation(const float *p, const int w)
{
  const int w2 = 2*w;
  const float av =
      (p[-w2-1] + p[-w2] + p[-w2+1] +
       p[-w-2]  + p[-w-1]  + p[-w]  + p[-w+1]  + p[-w+2] +
       p[-2]    + p[-1]    + p[0]   + p[1]     + p[2] +
       p[w-2]   + p[w-1]   + p[w]   + p[w+1]   + p[w+2] +
       p[w2-1]  + p[w2]  + p[w2+1]) / 21.0f;
  return sqrtf(
      (sqrf(p[-w2-1]-av) + sqrf(p[-w2]-av)  + sqrf(p[-w2+1]-av) +
       sqrf(p[-w-2]-av)  + sqrf(p[-w-1]-av) + sqrf(p[-w]-av)  + sqrf(p[-w+1]-av)  + sqrf(p[-w+2]-av) +
       sqrf(p[-2]-av)    + sqrf(p[-1]-av)   + sqrf(p[0]-av)   + sqrf(p[1]-av)     + sqrf(p[2]-av) +
       sqrf(p[w-2]-av)   + sqrf(p[w-1]-av)  + sqrf(p[w]-av)   + sqrf(p[w+1]-av)   + sqrf(p[w+2]-av) +
       sqrf(p[w2-1]-av)  + sqrf(p[w2]-av)   + sqrf(p[w2+1]-av)) / 21.0f);
}

static float _calc_weight(const float *s, const size_t loc, const int w, const float clipval)
{
  const float smoothness = fmaxf(0.0f, 1.0f - 10.0f * sqrtf(_local_std_deviation(&s[loc], w)));
  float val = 0.0f;
  for(int y = -1; y < 2; y++)
  {
    for(int x = -1; x < 2; x++)
      val += s[loc + y*w + x] / 9.0f;
  }
  const float sval = fmaxf(1.0f, powf(fminf(clipval, val) / clipval, 2.0f));
  return sval * smoothness;
}

static void _calc_plane_candidates(const float *plane,
                                   const float *refavg,
                                   dt_iop_segmentation_t *seg,
                                   const float clipval,
                                   const float badlevel)
{
  DT_OMP_PRAGMA(parallel for default(firstprivate) schedule(dynamic))
  for(uint32_t id = 2; id < seg->nr; id++)
  {
    seg->val1[id] = 0.0f;
    seg->val2[id] = 0.0f;
    // avoid very small segments
    if((seg->ymax[id] - seg->ymin[id] > 2) && (seg->xmax[id] - seg->xmin[id] > 2))
    {
      size_t testref = 0;
      float testweight = 0.0f;
      // make sure we don't calc a candidate from duplicated border data
      for(int row = MAX(seg->border+2, seg->ymin[id]-2); row < MIN(seg->height - seg->border-2, seg->ymax[id]+3); row++)
      {
        for(int col = MAX(seg->border+2, seg->xmin[id]-2); col < MIN(seg->width - seg->border-2, seg->xmax[id]+3); col++)
        {
          const size_t pos = row * seg->width + col;
          const uint32_t sid = _get_segment_id(seg, pos);
          if((sid == id) && (plane[pos] < clipval))
          {
            const float wht = _calc_weight(plane, pos, seg->width, clipval) * ((seg->data[pos] & DT_SEG_ID_MASK) ? 1.0f : 0.75f);
            if(wht > testweight)
            {
              testweight = wht;
              testref = pos;
            }
          }
        }
      }
      if(testref && (testweight > 1.0f - badlevel)) // We have found a reference location
      {
        float sum  = 0.0f;
        float pix = 0.0f;
        const float weights[5][5] = {
          { 1.0f,  4.0f,  6.0f,  4.0f, 1.0f },
          { 4.0f, 16.0f, 24.0f, 16.0f, 4.0f },
          { 6.0f, 24.0f, 36.0f, 24.0f, 6.0f },
          { 4.0f, 16.0f, 24.0f, 16.0f, 4.0f },
          { 1.0f,  4.0f,  6.0f,  4.0f, 1.0f }};
        for(int y = -2; y < 3; y++)
        {
          for(int x = -2; x < 3; x++)
          {
            const size_t pos = testref + y*seg->width + x;
            const gboolean unclipped = plane[pos] < clipval;
            sum += (unclipped) ? plane[pos] * weights[y+2][x+2] : 0.0f;
            pix += (unclipped) ? weights[y+2][x+2] : 0.0f;
          }
        }
        const float av = sum / fmaxf(1.0f, pix);
        if(av > 0.125f * clipval)
        {
          seg->val1[id] = fminf(clipval, av);
          seg->val2[id] = refavg[testref];
        }
      }
    }
  }
}

static void _initial_gradients(const size_t w,
                               const size_t height,
                               float *luminance,
                               float *distance,
                               float *gradient)
{
  darkroom_segbased_initial_gradients(luminance, distance, gradient, w, height);
}

static float _segment_maxdistance(float *distance,
                                  dt_iop_segmentation_t *seg,
                                  const uint32_t id)
{
  const int xmin = MAX(seg->xmin[id]-2, seg->border);
  const int xmax = MIN(seg->xmax[id]+3, seg->width - seg->border);
  const int ymin = MAX(seg->ymin[id]-2, seg->border);
  const int ymax = MIN(seg->ymax[id]+3, seg->height - seg->border);

  return darkroom_segbased_maxdistance(distance, seg->data, seg->width, seg->height,
                                       xmin, xmax, ymin, ymax, id);
}

static float _segment_attenuation(dt_iop_segmentation_t *seg, const uint32_t id, const int mode)
{
  const float attenuate[NUM_RECOVERY_MODES] = { 0.0f, 1.7f, 1.0f, 1.7f, 1.0f, 1.0f, 1.0f};
  if(mode < DT_RECOVERY_MODE_ADAPT)
    return attenuate[mode];
  else
  {
    const float maxdist = fmaxf(1.0f, seg->val1[id]);
    return fminf(1.7f, 0.9f + (3.0f / maxdist));
  }
}

static float _segment_correction(dt_iop_segmentation_t *seg,
                                 const uint32_t id,
                                 const int mode,
                                 const int recovery_close)
{
  const float correction = _segment_attenuation(seg, id, mode);
  return correction - 0.1f * (float)recovery_close;
}

static void _calc_distance_ring(const int xmin,
                                const int xmax,
                                const int ymin,
                                const int ymax,
                                float *gradient,
                                float *distance,
                                const float attenuate,
                                const float dist,
                                dt_iop_segmentation_t *seg,
                                const uint32_t id)
{
  darkroom_segbased_distance_ring(gradient, distance, seg->data,
                                  seg->width, seg->height,
                                  xmin, xmax, ymin, ymax, attenuate, dist, id);
}

static void _segment_gradients(float *distance,
                               float *gradient,
                               float *tmp,
                               const int mode,
                               dt_iop_segmentation_t *seg,
                               const uint32_t id,
                               const int recovery_close)
{
  const int xmin = MAX(seg->xmin[id]-1, seg->border);
  const int xmax = MIN(seg->xmax[id]+2, seg->width - seg->border);
  const int ymin = MAX(seg->ymin[id]-1, seg->border);
  const int ymax = MIN(seg->ymax[id]+2, seg->height - seg->border);
  const float attenuate = _segment_attenuation(seg, id, mode);
  const float strength = _segment_correction(seg, id, mode, recovery_close);

  float maxdist = 1.5f;
  while(maxdist < seg->val1[id])
  {
    _calc_distance_ring(xmin, xmax, ymin, ymax, gradient, distance, attenuate, maxdist, seg, id);
    maxdist += 1.5f;
  }

  if(maxdist > 4.0f)
  {
    darkroom_segbased_box_in(gradient, tmp, seg->width, xmin, xmax, ymin, ymax);
    dt_box_mean(tmp, ymax-ymin, xmax-xmin, 1, MIN((int)maxdist, 15), 2);
    darkroom_segbased_box_out(gradient, tmp, seg->data, seg->width, xmin, xmax, ymin, ymax, id);
  }
  darkroom_segbased_apply_strength(gradient, seg->data, seg->width, xmin, xmax, ymin, ymax, id, strength);
}

static void _add_poisson_noise(float *lum,
                               dt_iop_segmentation_t *seg,
                               const uint32_t id,
                               const float noise_level)
{
  const int xmin = MAX(seg->xmin[id], seg->border);
  const int xmax = MIN(seg->xmax[id]+1, seg->width - seg->border);
  const int ymin = MAX(seg->ymin[id], seg->border);
  const int ymax = MIN(seg->ymax[id]+1, seg->height - seg->border);
  uint32_t DT_ALIGNED_ARRAY state[4] = { splitmix32(ymin), splitmix32(xmin), splitmix32(1337), splitmix32(666) };
  xoshiro128plus(state);
  xoshiro128plus(state);
  xoshiro128plus(state);
  xoshiro128plus(state);
  for(int row = ymin; row < ymax; row++)
  {
    for(int col = xmin; col < xmax; col++)
    {
      const size_t v = (size_t)row * seg->width + col;
      if(seg->data[v] == id)
      {
        const float pnoise = poisson_noise(lum[v] * noise_level, noise_level, col & 1, state);
        lum[v] += pnoise;
      }
    }
  }
}

static void _masks_extend_border(float *const mask,
                                 const int width,
                                 const int height,
                                 const int border)
{
  darkroom_masks_extend_border(mask, width, height, border);
}

static void _process_segmentation(dt_dev_pixelpipe_iop_t *piece,
                                  const float *const input,
                                  float *const output,
                                  const dt_iop_roi_t *const roi_in,
                                  const dt_iop_roi_t *const roi_out,
                                  dt_iop_highlights_data_t *d,
                                  const int vmode,
                                  float *tmpout)
{
  const uint8_t(*const xtrans)[6] = (const uint8_t(*const)[6])piece->xtrans;
  const uint32_t filters = piece->filters;
  const gboolean fullpipe = dt_pipe_is_full(piece->pipe);
  const float clipval = MAX(0.1f, highlights_clip_magics[DT_IOP_HIGHLIGHTS_SEGMENTS] * d->clip);

  const dt_aligned_pixel_t icoeffs = { piece->pipe->dsc.temperature.coeffs[0], piece->pipe->dsc.temperature.coeffs[1], piece->pipe->dsc.temperature.coeffs[2]};
  const dt_aligned_pixel_t clips = { clipval * icoeffs[0], clipval * icoeffs[1], clipval * icoeffs[2]};
  const dt_aligned_pixel_t cube_coeffs = {cbrtf(clips[0]), cbrtf(clips[1]), cbrtf(clips[2]), 0.0f};

  const dt_dev_chroma_t *chr = &piece->module->dev->chroma;
  const gboolean late = chr->late_correction;
  const dt_aligned_pixel_t correction = { late ? (float)(chr->D65coeffs[0] / chr->as_shot[0]) : 1.0f,
                                          late ? (float)(chr->D65coeffs[1] / chr->as_shot[1]) : 1.0f,
                                          late ? (float)(chr->D65coeffs[2] / chr->as_shot[2]) : 1.0f,
                                          1.0f };
  const int recovery_mode = d->recovery;
  const float strength = d->strength;

  const int recovery_closing[NUM_RECOVERY_MODES] = { 0, 0, 0, 2, 2, 0, 2};
  const int recovery_close = recovery_closing[recovery_mode];
  const int segmentation_limit = (piece->pipe->iwidth * piece->pipe->iheight) * sqrf(piece->pipe->iscale) / 4000; // 250 segments per mpix

  const size_t pwidth  = dt_round_size(roi_in->width / 3, 2) + 2 * HL_BORDER;
  const size_t pheight = dt_round_size(roi_in->height / 3, 2) + 2 * HL_BORDER;
  const size_t p_size =  dt_round_size((size_t) pwidth * pheight, 64);

  float *fbuffer = dt_alloc_align_float(HL_FLOAT_PLANES * p_size);
  if(!fbuffer)
  {
    dt_print(DT_DEBUG_PIPE, "[process segmentation] can't allocate intermediate buffers");
    return;
  }

  float *plane[HL_FLOAT_PLANES];
  for(int i = 0; i < HL_FLOAT_PLANES; i++)
    plane[i] = fbuffer + i * p_size;

  float *refavg[HL_RGB_PLANES];
  for(int i = 0; i < HL_RGB_PLANES; i++)
    refavg[i] = plane[HL_SEGMENT_PLANES + i];

  gboolean segerror = FALSE;

  dt_iop_segmentation_t isegments[HL_SEGMENT_PLANES];
  for(int i = 0; i < HL_SEGMENT_PLANES; i++)
    segerror |= dt_segmentation_init_struct(&isegments[i], pwidth, pheight, HL_BORDER+1, segmentation_limit);

  if(segerror)
  {
    dt_print(DT_DEBUG_PIPE, "[process segmentation] can't allocate segmentation buffers");
    for(int i = 0; i < HL_SEGMENT_PLANES; i++)
      dt_segmentation_free_struct(&isegments[i]);

    dt_free_align(fbuffer);
    return;
  }

  const int xshifter = ((filters != 9u) && (FC(0, 0, filters) == 1)) ? 1 : 2;

  // populate the segmentation data, planes and refavg ...
  uint32_t *segdata[HL_SEGMENT_PLANES] = { isegments[0].data, isegments[1].data,
                                           isegments[2].data, isegments[3].data };
  int32_t has_allclipped_i = 0;
  const int32_t anyclipped = darkroom_segbased_populate_planes(
      tmpout, roi_in->width, roi_in->height, filters, (const unsigned char *)xtrans,
      correction, cube_coeffs, xshifter, plane, refavg, segdata,
      pwidth, pheight, &has_allclipped_i);
  const gboolean has_allclipped = (has_allclipped_i != 0);

  if((anyclipped < 20) && vmode == DT_HIGHLIGHTS_MASK_OFF)
    goto finish;

  for(int i = 0; i < HL_RGB_PLANES; i++)
    _masks_extend_border(plane[i], pwidth, pheight, HL_BORDER);

  for(int p = 0; p < HL_RGB_PLANES; p++)
    dt_segments_combine(&isegments[p], d->combine);

  if(dt_get_num_threads() >= HL_RGB_PLANES)
  {
    DT_OMP_PRAGMA(parallel num_threads(HL_RGB_PLANES))
    {
      dt_segmentize_plane(&isegments[dt_get_thread_num()]);
    }
  }
  else
  {
    for(int p = 0; p < HL_RGB_PLANES; p++)
      dt_segmentize_plane(&isegments[p]);
  }

  for(int p = 0; p < HL_RGB_PLANES; p++)
    _calc_plane_candidates(plane[p], refavg[p], &isegments[p], cube_coeffs[p], d->candidating);

  {
    const uint32_t *cdata[HL_RGB_PLANES] = { isegments[0].data, isegments[1].data, isegments[2].data };
    const float *cval1[HL_RGB_PLANES] = { isegments[0].val1, isegments[1].val1, isegments[2].val1 };
    const float *cval2[HL_RGB_PLANES] = { isegments[0].val2, isegments[1].val2, isegments[2].val2 };
    const int32_t cnr[HL_RGB_PLANES] = { isegments[0].nr, isegments[1].nr, isegments[2].nr };
    darkroom_segbased_candidates_apply(input, tmpout, roi_in->width, roi_in->height,
                                       filters, (const unsigned char *)xtrans, clips, correction,
                                       plane, cdata, cval1, cval2, cnr,
                                       pwidth, pheight, isegments[0].border);
  }

  float *distance  = plane[HL_RGB_PLANES];
  float *gradient  = plane[HL_RGB_PLANES + 1];
  float *luminance = plane[HL_RGB_PLANES + 2];
  float *recout    = plane[HL_RGB_PLANES + 3];
  float *tmp       = plane[HL_RGB_PLANES + 4];

  dt_iop_segmentation_t *segall = &isegments[3];

  const gboolean do_recovery = (recovery_mode != DT_RECOVERY_MODE_OFF) && has_allclipped && (strength > 0.0f);
  const gboolean do_masking = (vmode != DT_HIGHLIGHTS_MASK_OFF) && fullpipe;

  if(do_recovery || do_masking)
  {
    dt_segments_combine(segall, recovery_close);
    dt_iop_image_fill(gradient, fminf(1.0f, 5.0f * strength), pwidth, pheight, 1);
    dt_iop_image_fill(distance, 0.0f, pwidth, pheight, 1);
    darkroom_segbased_prepare_lumdist(plane[0], plane[1], plane[2], icoeffs,
                                      tmp, distance, segall->data,
                                      pwidth, pheight, segall->border);
    _masks_extend_border(tmp, pwidth, pheight, segall->border);
    dt_gaussian_fast_blur(tmp, luminance, pwidth, pheight, 1.2f, 0.0f, 20.0f, 1);
  }

  if(do_recovery)
  {
    const float max_distance = dt_image_distance_transform(NULL, distance, pwidth, pheight, 1.0f, DT_DISTANCE_TRANSFORM_NONE);
    if(max_distance > 3.0f)
    {
      dt_segmentize_plane(segall);
      _initial_gradients(pwidth, pheight, luminance, distance, recout);
      _masks_extend_border(recout, pwidth, pheight, segall->border);

      // now we check for significant all-clipped-segments and reconstruct data
      for(uint32_t id = 2; id < segall->nr; id++)
      {
        segall->val1[id] = _segment_maxdistance(distance, segall, id);

        if(segall->val1[id] > 2.0f)
          _segment_gradients(distance, recout, tmp, recovery_mode, segall, id, recovery_close);
      }

      dt_gaussian_fast_blur(recout, gradient, pwidth, pheight, 1.2f, 0.0f, 20.0f, 1);
      // possibly add some noise
      const float noise_level = d->noise_level;
      if(noise_level > 0.0f)
      {
        for(uint32_t id = 2; id < segall->nr; id++)
        {
          if(segall->val1[id] > 3.0f)
            _add_poisson_noise(gradient, segall, id, noise_level);
        }
      }

      const float dshift = 2.0f + (float)recovery_closing[recovery_mode];

      darkroom_segbased_apply_recovery(input, tmpout, roi_in->width, roi_in->height,
                                       filters, (const unsigned char *)xtrans, clips,
                                       distance, gradient, pwidth, pheight,
                                       strength, dshift);
    }
  }

  {
    const uint32_t *cdata[HL_RGB_PLANES] = { isegments[0].data, isegments[1].data, isegments[2].data };
    const float *cval1[HL_RGB_PLANES] = { isegments[0].val1, isegments[1].val1, isegments[2].val1 };
    const int32_t cnr[HL_RGB_PLANES] = { isegments[0].nr, isegments[1].nr, isegments[2].nr };
    darkroom_segbased_final_output(output, tmpout, luminance, gradient,
                                   roi_out->width, roi_out->height, roi_out->x, roi_out->y,
                                   roi_in->width, roi_in->height,
                                   filters, (const unsigned char *)xtrans,
                                   cdata, cval1, cnr, segall->data, segall->nr,
                                   pwidth, pheight, segall->border,
                                   do_masking, vmode, strength);
  }

  dt_print(DT_DEBUG_PERF, "[segmentation report %-12s] %5.1fMpix, segments: %3i red, %3i green, %3i blue, %3i all, %4i allowed",
      dt_dev_pixelpipe_type_to_str(piece->pipe->type),
      (float) (roi_in->width * roi_in->height) / 1.0e6f, isegments[0].nr -2, isegments[1].nr-2, isegments[2].nr-2, isegments[3].nr-2,
      segmentation_limit-2);

  finish:

  for(int i = 0; i < HL_SEGMENT_PLANES; i++)
    dt_segmentation_free_struct(&isegments[i]);
  dt_free_align(fbuffer);
}

