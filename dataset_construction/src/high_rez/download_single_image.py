import asyncio

async def download_and_validate_async(
        session,
        item_id,
        asset_type_id="ortho_visual",
        item_type_id="PSScene"
):
    cl = session.client('data')
  
    asset = await cl.get_asset(item_type_id, item_id, asset_type_id)
    await cl.activate_asset(asset)

    asset = await cl.wait_asset(asset, callback=print)
    path = await cl.download_asset(asset)
    cl.validate_checksum(asset, path)

    return path



