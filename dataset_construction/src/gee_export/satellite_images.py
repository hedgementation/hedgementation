import ee
import shapely

from src.gee_export.export_configs import normalize_date


def mosaic_by_date(img_collection):
    """Mosaic all images for the same date into a single image."""
    img_collection = img_collection.sort('system:time_start')
    
    dates = img_collection.aggregate_array('system:time_start').map(
        lambda t: ee.Date(t).format('YYYY-MM-dd')
    ).distinct()

    def mosaic_date(date_str):
        date = ee.Date(date_str)
        daily = img_collection.filterDate(date, date.advance(1, 'day'))
        return daily.mosaic().set('system:time_start', date.millis()).set('date', date_str)

    return ee.ImageCollection(dates.map(mosaic_date))

def shapely_to_ee(poly):
    """Convert Shapely geometry to EE geometry."""
    geo = poly.__geo_interface__
    return ee.Geometry(geo)


def stack_images(img_collection):
    """Stack multiple images into a single multi-band image."""
    img_collection = img_collection.sort('system:time_start')
    images_list = img_collection.toList(img_collection.size())
    first = ee.Image(images_list.get(0))
    rest = ee.List(images_list.slice(1))

    def stack(img, previous):
        return ee.Image(previous).addBands(ee.Image(img))

    return ee.Image(rest.iterate(stack, first))


def rename_bands(img):
    """Rename bands to include date information."""
    date = img.date().format('yyyy-MM-dd')
    band_names = img.bandNames().map(lambda x: ee.String(date).cat('_').cat(x))
    return img.rename(band_names)


def get_stacked_img_for_patch(geom, image_collection, filter_clouds=True, cloud_threshold=20):
    if filter_clouds:
        filtered_collection = image_collection.filterBounds(geom).filter(
            ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold)
        )
    else:
        filtered_collection = image_collection.filterBounds(geom)

    mosaicked = mosaic_by_date(filtered_collection)
    return stack_images(mosaicked.map(rename_bands))

def get_collection(collection_name: str, start_date: str, end_date: str, selected_bands: list):
    """Get an image collection filtered to the given date range and bands.

    Dates may be 'YYYY-MM-DD' or legacy 'YYYYMMDD'.
    """
    return ee.ImageCollection(collection_name).filter(
        ee.Filter.date(
            ee.Date(normalize_date(start_date)),
            ee.Date(normalize_date(end_date))
        )
    ).select(selected_bands)


def get_s2_date_string(geom, s2_collection, filter_cloudy, cloud_threshold):
    """
    Returns an ee.String containing comma-separated dates of images
    that pass the cloud filter for this specific geometry.
    """
    if filter_cloudy:
        filtered = s2_collection.filterBounds(geom).filter(
            ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold)
        )
    else:
        filtered = s2_collection.filterBounds(geom)
    date_list = filtered.aggregate_array('system:time_start').map(
        lambda t: ee.Date(t).format('YYYY-MM-dd')
    )
    return date_list.join(',')
