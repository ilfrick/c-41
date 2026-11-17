/* RGBE dimension checks shared by the decoder and its focused unit test. */
#ifndef DT_IMAGEIO_RGBE_VALIDATE_H
#define DT_IMAGEIO_RGBE_VALIDATE_H

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static inline int dt_imageio_rgbe_parse_int(const char **cursor, int *value)
{
  if(cursor == NULL || *cursor == NULL || value == NULL) return 0;

  char *end = NULL;
  errno = 0;
  const long parsed = strtol(*cursor, &end, 10);
  if(end == *cursor || errno == ERANGE || parsed < INT_MIN || parsed > INT_MAX)
    return 0;

  *value = (int)parsed;
  *cursor = end;
  return 1;
}

static inline int dt_imageio_rgbe_parse_dimensions_line(const char *line,
                                                         int *height,
                                                         int *width)
{
  if(line == NULL || height == NULL || width == NULL || line[0] != '-' || line[1] != 'Y')
    return 0;

  const char *cursor = line + 2;
  int parsed_height;
  int parsed_width;
  if(!dt_imageio_rgbe_parse_int(&cursor, &parsed_height)) return 0;

  while(isspace((unsigned char)*cursor)) cursor++;
  if(cursor[0] != '+' || cursor[1] != 'X') return 0;
  cursor += 2;

  if(!dt_imageio_rgbe_parse_int(&cursor, &parsed_width)) return 0;

  *height = parsed_height;
  *width = parsed_width;
  return 1;
}

static inline int dt_imageio_rgbe_float_allocation_fits(const size_t npixels)
{
  return npixels <= SIZE_MAX / (3 * sizeof(float));
}

static inline int dt_imageio_rgbe_validate_sizes(const size_t width,
                                                  const size_t height,
                                                  size_t *npixels)
{
  if(npixels == NULL || width == 0 || height == 0 || width > SIZE_MAX / height)
    return 0;

  const size_t pixels = width * height;
  if(pixels > INT_MAX || !dt_imageio_rgbe_float_allocation_fits(pixels))
    return 0;

  *npixels = pixels;
  return 1;
}

static inline int dt_imageio_rgbe_validate_dimensions(const int width,
                                                       const int height,
                                                       size_t *npixels)
{
  if(width <= 0 || height <= 0) return 0;

  return dt_imageio_rgbe_validate_sizes((size_t)width, (size_t)height, npixels);
}

#endif
