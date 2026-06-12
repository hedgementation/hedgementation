import ee


def get_multilines_hedgerows(feature):
    """Extract hedgerow geometry from a feature."""
    feature = ee.Feature(feature)
    hedgerows_patch = feature.get("hedgerows")
    return ee.Geometry.MultiLineString(ee.Dictionary(hedgerows_patch).get("coordinates"))


def rasterize_hedgerows(multiLineString,
                        crs,
                        bounding_rect,
                        width=3,
                        initial_scale=2,
                        final_scale=10,
                        reduce_resolution=True):
    """Rasterize hedgerow linestrings into a binary image."""
    crs = ee.Projection(crs)

    # Anchor the pixel grid to the patch boundary so exports are exactly nb_pixel×nb_pixel.
    # Using the global origin (0,0) causes a 129-pixel span whenever the patch boundary
    # doesn't fall on a grid line.
    bbox_coords = bounding_rect.bounds(maxError=1, proj=crs).coordinates().get(0)
    xs = ee.List(bbox_coords).map(lambda p: ee.List(p).get(0))
    ys = ee.List(bbox_coords).map(lambda p: ee.List(p).get(1))
    minx = xs.reduce(ee.Reducer.min())
    maxy = ys.reduce(ee.Reducer.max())

    buffered_linestring = multiLineString.buffer(width / 2)
    crs_transform_initial = [initial_scale, 0, minx, 0, -initial_scale, maxy]
    canvas = ee.Image.constant(0).reproject(
        crs=crs,
        crsTransform=crs_transform_initial
    ).clip(bounding_rect)

    raster = canvas.paint(buffered_linestring, color=1).clip(bounding_rect)

    crs_transform_final = [final_scale, 0, minx, 0, -final_scale, maxy]

    if reduce_resolution:
        return raster.reduceResolution(
            reducer=ee.Reducer.mean(),
            maxPixels=(final_scale / initial_scale) ** 2,
        ).reproject(crs=crs, crsTransform=crs_transform_final)
    else:
        return raster


def rasterize_hedgerows_identity(multiLineString,
                                  crs,
                                  bounding_rect,
                                  width=3,
                                  initial_scale=2,
                                  final_scale=10,
                                  reduce_resolution=True):
    """Rasterize hedgerow linestrings into an identity mask image.

    Each pixel holds the index of the linestring it overlaps with (0-indexed),
    or -1 if it overlaps with no linestring. At reduced resolution, each pixel
    is assigned the most common non-(-1) index among its constituent pixels,
    or -1 if all constituent pixels are background.
    """
    crs = ee.Projection(crs)

    bbox_coords = bounding_rect.bounds(maxError=1, proj=crs).coordinates().get(0)
    xs = ee.List(bbox_coords).map(lambda p: ee.List(p).get(0))
    ys = ee.List(bbox_coords).map(lambda p: ee.List(p).get(1))
    minx = xs.reduce(ee.Reducer.min())
    maxy = ys.reduce(ee.Reducer.max())

    crs_transform_initial = [initial_scale, 0, minx, 0, -initial_scale, maxy]
    canvas = ee.Image.constant(-1).toInt().reproject(
        crs=crs,
        crsTransform=crs_transform_initial
    ).clip(bounding_rect)

    geometries = multiLineString.geometries()
    n = geometries.size()

    def paint_linestring(i, image):
        i = ee.Number(i).toInt()
        image = ee.Image(image)
        geom = ee.Geometry(geometries.get(i))
        buffered = geom.buffer(width / 2)
        return image.paint(ee.FeatureCollection([ee.Feature(buffered)]), color=i)

    identity_raster = ee.Image(
        ee.List.sequence(0, n.subtract(1)).iterate(paint_linestring, canvas)
    ).clip(bounding_rect)

    crs_transform_final = [final_scale, 0, minx, 0, -final_scale, maxy]

    if reduce_resolution:
        # mode() ignores no values by default, so mask out -1 before reducing,
        # then restore -1 where the mask is empty after reduction.
        foreground_mask = identity_raster.neq(-1)
        masked = identity_raster.updateMask(foreground_mask)

        identity_reduced = masked.reduceResolution(
            reducer=ee.Reducer.mode(),
            maxPixels=(final_scale / initial_scale) ** 2,
        ).reproject(crs=crs, crsTransform=crs_transform_final)

        return identity_reduced.unmask(-1)
    else:
        return identity_raster


def rasterize_feature(feature,
                      crs,
                      width=3,
                      initial_scale=2,
                      final_scale=10,
                      reduce_resolution=True,
                      nb_pixel=128):
    """Rasterize a feature's hedgerows into a binary image."""
    feature = ee.Feature(feature)
    hedgerow_multiline = get_multilines_hedgerows(feature)
    bounding_rect = feature.geometry()
    result = rasterize_hedgerows(
        hedgerow_multiline,
        crs=crs,
        bounding_rect=bounding_rect,
        width=width,
        initial_scale=initial_scale,
        final_scale=final_scale,
        reduce_resolution=reduce_resolution
    )
    return result.set({"id": feature.get("id")})


def rasterize_feature_identity(feature,
                                crs,
                                width=3,
                                initial_scale=2,
                                final_scale=10,
                                reduce_resolution=True,
                                nb_pixel=128):
    """Rasterize a feature's hedgerows into an identity mask image.

    Each pixel holds the index of the overlapping linestring (0-indexed), or -1
    for background. See rasterize_hedgerows_identity for full semantics.
    """
    feature = ee.Feature(feature)
    hedgerow_multiline = get_multilines_hedgerows(feature)
    bounding_rect = feature.geometry()
    result = rasterize_hedgerows_identity(
        hedgerow_multiline,
        crs=crs,
        bounding_rect=bounding_rect,
        width=width,
        initial_scale=initial_scale,
        final_scale=final_scale,
        reduce_resolution=reduce_resolution
    )
    return result.set({"id": feature.get("id")})
