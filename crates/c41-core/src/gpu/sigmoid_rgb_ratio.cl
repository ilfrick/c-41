/*
 * Adapted from darktable data/kernels/sigmoid.cl,
 * sigmoid_loglogistic_rgb_ratio. Copyright (c) 2023-2026 darktable developers.
 * Licensed under GNU GPL-3.0-or-later; see the source file's license notice.
 * This kernel applies an ordered span of exposure and sigmoid stages to packed
 * RGBA frames. Frame transfer remains full-frame (no tiling).
 */
__kernel void sigmoid_rgb_ratio(
    __global const float *input,
    __global float *output,
    __global const float *stages,
    const ulong stage_count,
    const ulong pixel_count)
{
    const size_t pixel = get_global_id(0);
    if((ulong)pixel >= pixel_count) return;
    const size_t base = pixel * 4;
    float r = input[base];
    float g = input[base + 1];
    float b = input[base + 2];
    float alpha = input[base + 3];

    for(ulong index = 0; index < stage_count; ++index)
    {
        const size_t params = (size_t)index * 7;
        const float kind = stages[params];
        if(kind == 0.0f)
        {
            const float black = stages[params + 1];
            const float scale = stages[params + 2];
            r = (r - black) * scale;
            g = (g - black) * scale;
            b = (b - black) * scale;
            alpha = (alpha - black) * scale;
            continue;
        }

        const float white_target = stages[params + 1];
        const float black_target = stages[params + 2];
        const float paper_exp = stages[params + 3];
        const float film_fog = stages[params + 4];
        const float film_power = stages[params + 5];
        const float paper_power = stages[params + 6];
        const float average = fmax((r + g + b) / 3.0f, 0.0f);
        const float minimum = fmin(fmin(r, g), b);
        const float saturation = minimum < 0.0f ? -average / (minimum - average) : 1.0f;
        const float sr = average + saturation * (r - average);
        const float sg = average + saturation * (g - average);
        const float sb = average + saturation * (b - average);
        const float luma = (sr + sg + sb) / 3.0f;
        const float clamped = fmax(luma, 0.0f);
        const float film_response = pow(film_fog + clamped, film_power);
        const float response = white_target * pow(film_response / (paper_exp + film_response), paper_power);
        const float mapped = isnan(response) ? white_target : response;

        float pr, pg, pb;
        if(luma > 1.0e-9f) {
            const float scale = mapped / luma;
            pr = scale * sr; pg = scale * sg; pb = scale * sb;
        } else {
            pr = mapped; pg = mapped; pb = mapped;
        }
        const float pmin = fmin(fmin(pr, pg), pb);
        const float pmax = fmax(fmax(pr, pg), pb);
        const float epsilon = 1.0e-6f;
        const float dbvc_white = (white_target - mapped) / (pmax - mapped + epsilon);
        const float dbvc_black = (black_target - mapped) / (pmin - mapped - epsilon);
        const float dbvc = fmin(dbvc_white, dbvc_black);
        const float cvm = (mapped - pmin) / (mapped + epsilon);
        const float pca = 1.0f / (cvm * dbvc + epsilon);
        const float hyperbolic = 2.0f * cvm / (1.0f - cvm * cvm + epsilon) * pca;
        const float z = sqrt(hyperbolic * hyperbolic + 1.0f);
        const float factor = hyperbolic / (1.0f + z) * dbvc;
        r = mapped + factor * (pr - mapped);
        g = mapped + factor * (pg - mapped);
        b = mapped + factor * (pb - mapped);
    }
    output[base] = r;
    output[base + 1] = g;
    output[base + 2] = b;
    output[base + 3] = alpha;
}
