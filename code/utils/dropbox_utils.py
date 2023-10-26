import os
import dropbox


def init_dropbox(token_path):
    with open(token_path) as f:
        token = f.readlines()[0]
    dbx = dropbox.Dropbox(token)
    
    return dbx

def get_shared_url(path, dbx):

    # result = dbx.sharing_get_shared_links(path)
    result = dbx.sharing_list_shared_links(path)
    # the above command was changed sep 2023 because something changed with dropbox
    # sharing link formats
    if len(result.links)==0:
        meta = dbx.sharing_create_shared_link_with_settings(path)
        # result = dbx.sharing_get_shared_links(path)
        result = dbx.sharing_list_shared_links(path)
    url = result.links[0].url
    url = url.replace('www.dropbox', 'dl.dropboxusercontent')
    
    return url