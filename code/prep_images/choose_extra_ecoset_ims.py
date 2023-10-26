import sys, os
import numpy as np
import time
from sklearn import decomposition
import pandas as pd
import torch

import scipy.spatial.distance


project_root = '/user_data/mmhender/featsynth/'
ecoset_info_path = '/user_data/mmhender/stimuli/ecoset_info/'

def choose_best_ecoset_ims():

    # choosing a set of ecoset images to augment the "things" images
    # based on which ecoset images have most similar clip embeddings to 
    # the original things images. 
    
    feat_path = os.path.join(project_root, 'features', 'clip')

    
    image_set_name1 = 'images_things64'
    image_set_name2 = 'images_ecoset64'

    # load clip embeddings for things and ecoset
    feat_file_name1 = os.path.join(feat_path,'%s_clip_embed.npy'%(image_set_name1))
    print(feat_file_name1)
    e_things = np.load(feat_file_name1)
    # load corresponding labels
    image_list_filename1 = os.path.join(project_root, 'features','raw', '%s_list.csv'%(image_set_name1))
    print(image_list_filename1)
    labels_things = pd.read_csv(image_list_filename1, index_col=0)
    labels_things['orig_set']='things'

    feat_file_name2 = os.path.join(feat_path,'%s_clip_embed.npy'%(image_set_name2))
    print(feat_file_name2)
    e_ecoset = np.load(feat_file_name2)
    image_list_filename2 = os.path.join(project_root, 'features','raw', '%s_list.csv'%(image_set_name2))
    print(image_list_filename2)
    labels_ecoset = pd.read_csv(image_list_filename2, index_col=0)
    labels_ecoset['orig_set']='ecoset'
    
    # info about ecoset categories
    fn = os.path.join(ecoset_info_path, 'categ_use_ecoset.npy')
    info = np.load(fn, allow_pickle=True).item()
    bnames = list(info['binfo'].keys())
    
    
    ecoset_ims_exclude = exclude_ims()

    
    labels_all = pd.DataFrame()
    
    # how many total ims do we want per categ?
    # n_total = 61
    n_total = 40
    n_things = 12
    n_add_ecoset = n_total - n_things
    
    for bi, bname in enumerate(bnames):
    
        # print(bname)
        inds_things = np.where(labels_things['basic_name']==bname)[0]

        inds_ecoset = np.where(labels_ecoset['basic_name']==bname)[0]
        
        
        # cosine distances from each things image to each ecoset image
        dist_each = scipy.spatial.distance.cdist(e_things[inds_things,:], \
                                                 e_ecoset[inds_ecoset,:], metric='cosine')
        
        # trying to filter out duplicates here - if they have very small cosine dist
        # to any of the things images. this actually happens because some things 
        # images are taken from ecoset, but the names changed.
        close_to_any_things = np.any(dist_each<0.025, axis=0)
        if np.any(close_to_any_things):
            print('%s: detected %d duplicates with things, skipping these'%(bname, np.sum(close_to_any_things)))
        
        # also find images that have a duplicate within the ecoset images only. this happens sometimes
        dist_within_ecoset = scipy.spatial.distance.squareform(\
                        scipy.spatial.distance.pdist(e_ecoset[inds_ecoset,:], metric='cosine'))
        # this is very approximate cutoff for cosine distance, still need to check everything manually
        # dups_within = np.where((dist_within_ecoset>0) & (dist_within_ecoset<0.025))
        dups_within = np.where((dist_within_ecoset<0.025))
        if len(dups_within[0])>0:
            # the matrix is symmetric so get rid of redundant pairs
            dup_pairs = list(zip(dups_within[0], dups_within[1]))
            dups_keep = [dup for dup in dup_pairs if dup[1]>dup[0]]
            # ditch the second of each duplicate pair
            dup_inds = [dup[1] for dup in dups_keep]
            dup_within_ecoset = np.isin(np.arange(len(inds_ecoset)), dup_inds)
            if np.any(dup_within_ecoset):
                print('%s: detected %d duplicates within ecoset, skipping these'%(bname, np.sum(dup_within_ecoset)))
        else:
            dup_within_ecoset = np.zeros((len(inds_ecoset),),dtype=bool)
            
        # also exclude any that we manually rejected
        ecoset_filenames = np.array(labels_ecoset.iloc[inds_ecoset]['image_filename'])
        ecoset_filenames = np.array([i.split('Ecoset/')[1] for i in ecoset_filenames])
        skip_ims = np.isin(ecoset_filenames, ecoset_ims_exclude)
        if np.any(skip_ims):
            print('%s: detected %d bad images, skipping these'%(bname, np.sum(skip_ims)))
        
        exclude = close_to_any_things | skip_ims | dup_within_ecoset
        
        if bname=='trumpet':
            # only using the hand-picked ones here
            good_names = get_trumpet_filenames()
            not_good = ~np.isin(ecoset_filenames, good_names)
            print('%d good trumpet images'%len(good_names))
            print(np.sum(not_good))
            print(np.sum(exclude))
            exclude = exclude | not_good
            print(np.sum(exclude))
            print(np.mean(exclude))
            
        # get average dist over the things images
        dist_each = np.mean(dist_each, axis=0)
        
        # want to ignore some images, set to big value
        dist_each[exclude] = np.max(dist_each)+100
        
        # picking ecoset images w min distance
        best_inds = inds_ecoset[np.argsort(dist_each)][0:n_add_ecoset]
        
        # putting these all into a new dataframe
        labels_new = pd.DataFrame()
        
        labels_new = pd.concat([labels_new, labels_things.iloc[inds_things]])
        
        labels_new = pd.concat([labels_new, labels_ecoset.iloc[best_inds]])

        labels_new['exemplar_number_orig']=labels_new['exemplar_number']
        labels_new['exemplar_number'] = np.arange(n_total)
        
        all_filenames = list(labels_new['image_filename'])
        
        # making sure we successfully excluded bad images
        ecoset_filenames = np.array([i.split('Ecoset/')[1] if 'Ecoset' in i else i.split('things/')[1] \
                             for i in all_filenames])
        bad = np.isin(ecoset_filenames, ecoset_ims_exclude)
        assert(not np.any(bad))
        
        # trying to screen for any duplicate files here
        simple_filenames = [f.split('/')[-1] for f in all_filenames]
        assert len(simple_filenames)==len(np.unique(simple_filenames))
        # print(len(simple_filenames), len(np.unique(simple_filenames)))
        
        # duplicates would have very small cosine dist between them
        all_embeds = np.concatenate([e_things[inds_things,:], \
                                     e_ecoset[best_inds,:]], axis=0)
        dist_each = scipy.spatial.distance.pdist(all_embeds, metric='cosine')
        assert(np.all(dist_each>=0.010))
        

    
        labels_all = pd.concat([labels_all, labels_new])
    
    
    # save all the labels to a new big list
    labels_all = labels_all.set_index(np.arange(labels_all.shape[0]))

    image_set_name_save = 'images_comb64'
        
    image_list_name_save = os.path.join(project_root, 'features','raw', '%s_list.csv'%(image_set_name_save))
    print('writing to %s'%image_list_name_save)
    
    labels_all.to_csv(image_list_name_save)


def exclude_ims():

    """ these are a list of any images that have any issues.
    
    # reasons for excluding are:
    # if they have something like an artificial border or graphic that makes it look unnatural. 
    # if it is a duplicate of things image with slightly different crop
    # if background is very plain white with no context
    # if there are other objects prominent, esp one of the other categories used
    # similar to another item in the ecoset images for same category, in idiosyncratic way
    # (like they are obviously on same distinct background)
    # some where the center cropping removes a large chunk of the important object 
    # (this is mostly for the tools category like hammer). 
    """
    
    ecoset_ims_exclude = ['train/0593_butterfly/n02279972_22284.JPEG',\
                          'train/0593_butterfly/n02279972_28215.JPEG',\
                          'train/1076_ant/n02220518_4413.JPEG', \
                         'train/2268_moth/n02292401_2704.JPEG', \
                         'train/2268_moth/n02295064_3439.JPEG', \
                         'train/1021_mosquito/n02200850_130.JPEG', \
                         'train/1021_mosquito/n02200850_5048.JPEG', \
                         'train/1021_mosquito/n02201626_1985.JPEG', \
                         'train/1021_mosquito/n02200850_2225.JPEG', \
                         'train/1021_mosquito/n02201626_962.JPEG', \
                         'train/1021_mosquito/n02202006_532.JPEG', \
                         'train/1021_mosquito/n02200850_4081.JPEG', \
                         'train/1021_mosquito/n02201000_438.JPEG', \
                          'train/1021_mosquito/n02201000_6212.JPEG',
                          'train/1021_mosquito/n02201000_4888.JPEG',
                          'train/1021_mosquito/n02202006_2153.JPEG',
                          'train/1021_mosquito/n02200198_10792.JPEG',
                          'train/0828_pea/n12560282_17493.JPEG',
                          'train/0828_pea/n12561594_1804.JPEG',
                          'train/0828_pea/n07726386_2729.JPEG',
                          'train/1096_onion/n07722217_14237.JPEG', \
                          'train/1884_cabbage/n11875691_6497.JPEG', \
                          'train/1884_cabbage/n07713895_8099.JPEG', \
                          'train/1884_cabbage/n07714571_6198.JPEG',
                          'train/0387_lettuce/n07724269_14476.JPEG', \
                          'train/0387_lettuce/n11986511_13549.JPEG', \
                          'train/1563_beet/n07719839_18823.JPEG', \
                          'train/1563_beet/n07719839_27666.JPEG', \
                          'train/0506_asparagus/n12441183_1138.JPEG', \
                          'train/0506_asparagus/n12441183_5576.JPEG', \
                          'train/0222_grape/n13145040_9634.JPEG', \
                          'train/2569_cherry/n07757312_9710.JPEG', \
                          'train/2569_cherry/n07757990_1400.JPEG',
                          'train/0643_raspberry/n12656528_3587.JPEG', \
                          'train/0643_raspberry/n07745466_9864.JPEG',
                          'train/0115_apple/n07739506_2869.JPEG', \
                          'train/0115_apple/n07739125_3049.JPEG', \
                          'train/0115_apple/n07739506_8488.JPEG', \
                          'train/0115_apple/n07739125_3073.JPEG', \
                          'train/0115_apple/n07739125_10797.JPEG', \
                          'train/0115_apple/n12633638_8371.JPEG', \
                          'train/0665_pear/n07767847_6233.JPEG', \
                          'train/0170_banana/n12353203_2566.JPEG', \
                          'train/0170_banana/n12353203_4328.JPEG', \
                          'train/0170_banana/n07753592_12252.JPEG', \
                          'train/0170_banana/n12353203_2507.JPEG', \
                          'train/0170_banana/n12353203_4223.JPEG', \
                          'train/0170_banana/n07753592_7688.JPEG', \
                          'train/0170_banana/n07753592_5734.JPEG', \
                          'train/0170_banana/n07753592_6861.JPEG', \
                          'train/0170_banana/n07753592_9454.JPEG', \
                          'train/0170_banana/n07753592_9404.JPEG', \
                          'train/0170_banana/n07753592_7845.JPEG',
                          'train/1202_pomegranate/n07768694_26457.JPEG', \
                          'train/1202_pomegranate/n07768694_9717.JPEG', \
                          'train/1202_pomegranate/n12345280_16166.JPEG', \
                          'train/1202_pomegranate/n07768694_5658.JPEG', \
                          'train/1202_pomegranate/n11865276_1196.JPEG', \
                          'train/1202_pomegranate/n12345280_6049.JPEG', \
                          'train/1551_coconut/n07772935_6862.JPEG', \
                          'train/1551_coconut/n07772935_5145.JPEG', \
                          'train/1551_coconut/n07772935_2108.JPEG', \
                          'train/1551_coconut/n07772935_5921.JPEG', \
                          'train/1551_coconut/n07772935_5152.JPEG', \
                          'train/1551_coconut/n07772935_5756.JPEG', \
                          'train/1551_coconut/n07772935_13518.JPEG', \
                          'train/1551_coconut/n07772935_56.JPEG', \
                          'train/0406_pastry/flickr_pastry_0037.jpg', \
                          'train/0406_pastry/n07693590_13325.JPEG',
                          'train/0406_pastry/bing_pastry_0038.jpg',
                          'train/0229_cupcake/bing_cupcake_0004.jpg', \
                          'train/0229_cupcake/flickr_cupcake_0329.jpg',
                          'train/0141_cookie/bing_cookie_0010.jpg', \
                          'train/0335_pie/flickr_pie_0262.jpg', \
                          'train/0429_milkshake/bing_milkshake_0519.jpg', \
                          'train/0429_milkshake/bing_milkshake_0022.jpg', \
                          'train/0429_milkshake/bing_milkshake_0174.jpg',
                          'train/1409_brownie/flickr_brownie_0197.jpg', \
                          'train/1409_brownie/bing_brownie_0465.jpg', \
                          'train/1409_brownie/bing_brownie_0323.jpg', \
                          'train/1409_brownie/bing_brownie_0151.jpg', \
                          'train/1409_brownie/bing_brownie_0119.jpg', \
                          'train/1409_brownie/bing_brownie_0263.jpg', \
                          'train/1409_brownie/bing_brownie_0252.jpg',
                          'train/0846_pencil/n03908204_6640.JPEG', \
                          'train/0846_pencil/n03652100_7068.JPEG', \
                          'train/0846_pencil/n03908204_3873.JPEG', \
                          'train/0846_pencil/n13863020_24564.JPEG', \
                          'train/0846_pencil/n03908204_19902.JPEG', \
                          'train/0174_knife/n02973904_4057.JPEG', \
                          'train/0174_knife/n02893941_1460.JPEG', \
                          'train/0174_knife/n03549473_6246.JPEG', \
                          'train/0174_knife/n03890093_749.JPEG', \
                          'train/0174_knife/n04373089_11880.JPEG', \
                          'train/0174_knife/n02893941_476.JPEG',\
                          'train/0174_knife/n03890093_558.JPEG', \
                          'train/0174_knife/n04373089_146.JPEG', \
                          'train/0174_knife/n03624400_2362.JPEG', \
                          'train/0174_knife/n03623556_961.JPEG', \
                          'train/0174_knife/n03624400_2721.JPEG', \
                          'train/0174_knife/n03549473_3885.JPEG', \
                          'train/0174_knife/n03623556_2118.JPEG', \
                          'train/0174_knife/n03624400_5270.JPEG', \
                          'train/0174_knife/n03890093_2903.JPEG', \
                          'train/0174_knife/n03549473_6889.JPEG',
                          'train/0174_knife/n02927053_2484.JPEG',
                          'train/0174_knife/n03624400_2421.JPEG',
                          'train/0174_knife/n02976249_1998.JPEG',
                          'train/0174_knife/n02976249_1998.JPEG',
                          'train/0174_knife/n02976249_3183.JPEG',
                          'train/0174_knife/n04373089_3459.JPEG',
                          'train/0215_axe/n02764398_3585.JPEG', \
                          'train/0215_axe/n02764505_1803.JPEG', \
                          'train/0215_axe/n02764505_11319.JPEG',
                          'train/0215_axe/n02811468_15157.JPEG',
                          'train/0215_axe/n02764398_3598.JPEG',
                          'train/0215_axe/n02764398_1177.JPEG',
                          'train/0215_axe/n02764044_24763.JPEG',
                          'train/0215_axe/n02764398_4409.JPEG',
                          'train/0215_axe/n02764398_336.JPEG',
                          'train/0215_axe/n02764505_3304.JPEG',
                          'train/0215_axe/n02764398_1814.JPEG',
                          'train/0215_axe/n02764398_5224.JPEG',
                          'train/0215_axe/n02764398_2844.JPEG',
                          'train/0215_axe/n02764044_34214.JPEG',
                          'train/0215_axe/n02811468_15064.JPEG',
                          'train/0215_axe/n02764398_530.JPEG',
                          'train/0215_axe/n02764044_12336.JPEG',
                          'train/0215_axe/n02764044_2586.JPEG',
                          'train/0215_axe/n03077442_2184.JPEG',
                          'train/0215_axe/n02764505_6411.JPEG',
                          'train/0215_axe/n02764505_2215.JPEG',
                          'train/0215_axe/n02764505_1171.JPEG',
                          'train/0215_axe/n02764398_5046.JPEG',
                          'train/0215_axe/n02764505_6275.JPEG',
                          'train/0215_axe/n02764398_4209.JPEG',
                          'train/0215_axe/n02764398_1147.JPEG',
                          'train/0215_axe/n02764398_3999.JPEG',
                          'train/0215_axe/n02764044_23780.JPEG',
                          'train/0215_axe/n02764398_1915.JPEG',
                          'train/0215_axe/n02811468_13087.JPEG',
                          'train/0215_axe/n02764398_1499.JPEG',
                          'train/0215_axe/n02764044_29537.JPEG',
                          'train/0215_axe/n02764044_20847.JPEG',
                          'train/0215_axe/n02764044_1227.JPEG',
                          'train/0215_axe/n02764398_5085.JPEG',
                          'train/0215_axe/n02764398_5038.JPEG',
                          'train/0215_axe/n02764505_4438.JPEG', 
                          'train/0875_broom/n02907082_12807.JPEG',
                          'train/0875_broom/n02906734_609.JPEG',
                          'train/0875_broom/n04026918_1761.JPEG',
                          'train/0875_broom/n02906734_900.JPEG',
                          'train/0875_broom/n04026918_894.JPEG',
                          'train/0875_broom/n02906734_19240.JPEG',
                          'train/0875_broom/n02906734_4655.JPEG',
                          'train/0875_broom/n02906734_21374.JPEG', \
                          'train/0875_broom/n02907082_18709.JPEG',
                          'train/0875_broom/n02831894_6875.JPEG',
                          'train/0875_broom/n02831894_2112.JPEG',
                          'train/0875_broom/n02831894_776.JPEG',
                          'train/1543_hammer/n03481521_15587.JPEG', \
                          'train/1543_hammer/n03481172_5082.JPEG',
                          'train/1543_hammer/n03481172_25052.JPEG',
                          'train/1543_hammer/n03482001_4098.JPEG',
                          'train/1543_hammer/n03481172_18967.JPEG',
                          'train/1543_hammer/n03481172_4034.JPEG',
                          'train/1543_hammer/n03715669_14124.JPEG',
                          'train/1543_hammer/n02966545_8569.JPEG',
                          'train/1543_hammer/n02966545_4388.JPEG',
                          'train/1543_hammer/n03481521_13471.JPEG',
                          'train/1543_hammer/n04383301_847.JPEG',
                          'train/1543_hammer/n03715669_22104.JPEG',
                          'train/1543_hammer/n02783035_562.JPEG',
                          'train/1543_hammer/n03481172_14084.JPEG',
                          'train/1543_hammer/n02898173_508.JPEG',
                          'train/1543_hammer/n03481521_15888.JPEG',
                          'train/1543_hammer/n02783035_372.JPEG',
                          'train/1543_hammer/n02966545_3451.JPEG',
                          'train/1543_hammer/n03481172_3881.JPEG',
                          'train/1543_hammer/n02966545_4355.JPEG',
                          'train/1543_hammer/n03482001_6462.JPEG',
                          'train/1543_hammer/n03715669_6442.JPEG',
                          'train/1543_hammer/n03482001_4496.JPEG',
                          'train/1543_hammer/n03481172_4219.JPEG',
                          'train/1543_hammer/n03481172_17106.JPEG',
                          'train/1543_hammer/n03481172_9905.JPEG',
                          'train/1543_hammer/n03481521_8641.JPEG',
                          'train/1543_hammer/n02783035_855.JPEG',
                          'train/1543_hammer/n03481172_14739.JPEG',
                          'train/1543_hammer/n02966545_2853.JPEG',
                          'train/1543_hammer/n03715669_17290.JPEG',
                          'train/1543_hammer/n03481172_12077.JPEG',
                          'train/1543_hammer/n03481172_29830.JPEG', 
                          'train/1543_hammer/n03715669_6115.JPEG',
                          'train/1543_hammer/n03715669_12552.JPEG',
                          'train/1543_hammer/n03481521_7639.JPEG', 
                          'train/1543_hammer/n03481172_16227.JPEG',
                          'train/1543_hammer/n03481172_482.JPEG',
                          'train/1543_hammer/n03481521_9389.JPEG',
                          'train/1543_hammer/n03481172_1272.JPEG',
                          'train/1543_hammer/n02783035_727.JPEG',
                          'train/1543_hammer/n03715669_24775.JPEG',
                          'train/1543_hammer/n03715669_8928.JPEG',
                          'train/1543_hammer/n03481172_32788.JPEG',
                          'train/1543_hammer/n02966545_9155.JPEG',
                          'train/0359_shovel/n04208427_7652.JPEG',
                          'train/0359_shovel/n03418158_1315.JPEG',
                          'train/0359_shovel/n04208210_1871.JPEG',
                          'train/0359_shovel/n04208427_3966.JPEG',
                          'train/0359_shovel/n04208210_18570.JPEG',
                          'train/0359_shovel/n03418158_1957.JPEG',
                          'train/0359_shovel/n03418158_127.JPEG',
                          'train/0359_shovel/n03488603_2910.JPEG',
                          'train/0359_shovel/n04208210_23130.JPEG',
                          'train/0359_shovel/n03488603_3440.JPEG',
                          'train/0359_shovel/n04208210_9234.JPEG',
                          'train/0359_shovel/n04208210_17946.JPEG',
                          'train/0359_shovel/n03418158_772.JPEG',
                          'train/0359_shovel/n04208427_10606.JPEG',
                          'train/0359_shovel/n03214450_934.JPEG',
                          'train/0359_shovel/n03418158_1532.JPEG',
                          'train/0359_shovel/n04208427_3563.JPEG',
                          'train/0359_shovel/n03418158_3969.JPEG',
                          'train/0359_shovel/n03418158_3239.JPEG',
                          'train/0359_shovel/n04208427_11174.JPEG',
                          'train/0359_shovel/n03488603_2662.JPEG',
                          'train/0359_shovel/n04208427_3357.JPEG',
                          'train/0359_shovel/n04208210_15989.JPEG', 
                          'train/0359_shovel/n03488603_2347.JPEG', 
                          'train/0359_shovel/n04208210_3935.JPEG', 
                          'train/0359_shovel/n04208210_11998.JPEG', 
                          'train/0359_shovel/n04208427_1925.JPEG',
                          'train/0359_shovel/n04208210_6369.JPEG',
                          'train/0359_shovel/n03418158_18.JPEG',
                          'train/0359_shovel/n04208427_9398.JPEG',
                          'train/0359_shovel/n03418158_1002.JPEG',
                          'train/0359_shovel/n03418158_3479.JPEG',
                          'train/0359_shovel/n03418158_2465.JPEG',
                          'train/0359_shovel/n03418158_2686.JPEG', 
                          'train/0359_shovel/n04208427_2023.JPEG',
                          'train/0383_spoon/n03557270_2052.JPEG',
                          'train/0383_spoon/n04284002_1151.JPEG',
                          'train/0383_spoon/n03180384_4028.JPEG',
                          'train/0383_spoon/n04381073_1923.JPEG',
                          'train/0383_spoon/n04284002_12661.JPEG',
                          'train/0383_spoon/n04119630_129.JPEG',
                          'train/0383_spoon/n04398688_7523.JPEG', 
                          'train/0383_spoon/n03180384_4392.JPEG', 
                          'train/0383_spoon/n04284002_12400.JPEG',
                          'train/0383_spoon/n04381073_11801.JPEG',
                          'train/0383_spoon/n03180384_6387.JPEG',
                          'train/0383_spoon/n04381073_4103.JPEG',
                          'train/0383_spoon/n03180384_6731.JPEG',
                          'train/0383_spoon/n03180384_4291.JPEG',
                          'train/0383_spoon/n03557270_355.JPEG',
                          'train/0383_spoon/n03180384_3630.JPEG',
                          'train/0383_spoon/n04263502_6637.JPEG',
                          'train/0383_spoon/n03557270_1515.JPEG',
                          'train/0383_spoon/n03557270_1648.JPEG',
                          'train/0383_spoon/n04398688_12923.JPEG',
                          'train/0383_spoon/n04284002_15007.JPEG',
                          'train/0383_spoon/n04381073_3275.JPEG',
                          'train/1123_scissors/n04148054_7303.JPEG',
                          'train/1123_scissors/n04148054_1701.JPEG',
                          'train/1123_scissors/n04186848_19244.JPEG',
                          'train/1123_scissors/n04148054_7158.JPEG',
                          'train/1123_scissors/n04148054_1806.JPEG',
                          'train/1123_scissors/n04148054_1067.JPEG',
                          'train/1123_scissors/n04148054_16430.JPEG',
                          'train/1123_scissors/n04250473_1336.JPEG',
                          'train/1123_scissors/n04016684_1760.JPEG',
                          'train/1123_scissors/n04016684_4077.JPEG',
                          'train/1123_scissors/n03045074_5839.JPEG',
                          'train/1123_scissors/n04424692_1204.JPEG',
                          'train/1123_scissors/n04148054_2617.JPEG',
                          'train/1123_scissors/n04016684_4727.JPEG',
                          'train/1123_scissors/n04186848_6094.JPEG',
                          'train/1123_scissors/n04148054_6792.JPEG',
                          'train/1123_scissors/n04424692_2140.JPEG',
                          'train/1123_scissors/n04186848_3688.JPEG',
                          'train/1123_scissors/n04148054_7721.JPEG',
                          'train/1123_scissors/n04148054_20064.JPEG',
                          'train/1123_scissors/n04148054_17761.JPEG',
                          'train/1123_scissors/n04424692_1994.JPEG',
                          'train/1123_scissors/n04250473_1791.JPEG',
                          'train/1123_scissors/n04250473_11846.JPEG',
                          'train/1123_scissors/n04148054_514.JPEG',
                          'train/1123_scissors/n04148054_16662.JPEG',
                          'train/1123_scissors/n04186848_20907.JPEG',
                          'train/1123_scissors/n04186848_21498.JPEG',
                          'train/1123_scissors/n04186848_7736.JPEG',
                          'train/1123_scissors/n04148054_4002.JPEG',
                          'train/1123_scissors/n04250473_6199.JPEG',
                          'train/1123_scissors/n04250473_3177.JPEG',
                          'train/1123_scissors/n04186848_8173.JPEG',
                          'train/1123_scissors/n04186848_2617.JPEG',
                          'train/1123_scissors/n04148054_440.JPEG',
                          'train/1123_scissors/n04016684_4150.JPEG',
                          'train/1123_scissors/n04016684_4575.JPEG',
                          'train/1123_scissors/n04016684_2031.JPEG',
                          'train/1123_scissors/n04148054_2060.JPEG',
                          'train/1123_scissors/n04424692_1497.JPEG',
                          'train/1123_scissors/n04186848_3135.JPEG',
                          'train/1123_scissors/n04148054_21200.JPEG',
                          'train/1123_scissors/n04148054_291.JPEG', 
                          'train/1123_scissors/n04186848_21737.JPEG',
                          'train/1123_scissors/n04148054_1854.JPEG',
                          'train/1123_scissors/n04186848_3016.JPEG',
                          'train/1123_scissors/n04186848_17218.JPEG',
                          'train/1123_scissors/n04148054_7765.JPEG',
                          'train/1123_scissors/n04148054_1338.JPEG',
                          'train/0112_bell/n03028596_5156.JPEG', 
                          'train/0602_guitar/n02676566_5866.JPEG',
                          'train/0602_guitar/n02676566_868.JPEG',
                          'train/0602_guitar/n02676566_222.JPEG', 
                          'train/0602_guitar/n03467517_7233.JPEG',
                          'train/0602_guitar/n02676566_7320.JPEG',
                          'train/0602_guitar/n02676566_8525.JPEG',
                          'train/0375_drum/n03249569_19455.JPEG', 
                          'train/0375_drum/n03249569_31822.JPEG',
                          'train/0375_drum/n04249415_9539.JPEG',
                          'train/0375_drum/n04249415_6299.JPEG',
                          'train/0375_drum/n03249569_17598.JPEG',
                          'train/0375_drum/n02803666_739.JPEG',
                          'train/0375_drum/n04249415_442.JPEG',
                          'train/0375_drum/n02803666_6847.JPEG',
                          'train/0423_violin/n02700895_4881.JPEG',
                          'train/0423_violin/n04330998_7705.JPEG',
                          'train/0423_violin/n03465500_998.JPEG',
                          'train/0423_violin/n03465500_575.JPEG',
                          'train/0423_violin/n02700895_888.JPEG',
                          'train/1307_bugle/n03110669_129909.JPEG',
                          'train/0754_clarinet/n02803539_7153.JPEG',
                          'train/0754_clarinet/n03037709_13450.JPEG',
                          'train/0754_clarinet/n02803539_4100.JPEG',
                          'train/0754_clarinet/n03037709_26245.JPEG',
                          'train/0754_clarinet/n02803539_1992.JPEG',
                          'train/0754_clarinet/n02803539_5426.JPEG',
                          'train/0754_clarinet/n02803539_21.JPEG',
                          'train/0754_clarinet/n03037709_18331.JPEG',
                          'train/0754_clarinet/n02803539_4700.JPEG',
                          'train/0754_clarinet/n02803539_6600.JPEG',
                          'train/0754_clarinet/n02803539_3116.JPEG',
                          'train/0754_clarinet/n02803809_2346.JPEG',
                          'train/0754_clarinet/n03037709_8254.JPEG',
                          'train/0754_clarinet/n02803809_2391.JPEG',
                          'train/0754_clarinet/n03037709_14250.JPEG',
                          'train/0754_clarinet/n02803539_6534.JPEG',
                          'train/0754_clarinet/n02803539_4386.JPEG',
                          'train/0754_clarinet/n03037709_1054.JPEG',
                          'train/0754_clarinet/n02803539_8673.JPEG',
                          'train/0754_clarinet/n02803539_3370.JPEG',
                          'train/0754_clarinet/n03037709_13576.JPEG',
                          'train/0754_clarinet/n03037709_3360.JPEG', 
                          'train/0754_clarinet/n03037709_18061.JPEG',
                          'train/0754_clarinet/n03037709_8085.JPEG',
                          'train/0754_clarinet/n03037709_1083.JPEG',
                          'train/0754_clarinet/n03037709_6406.JPEG',
                          'train/0754_clarinet/n03037709_22762.JPEG',
                          'train/0754_clarinet/n03037709_18605.JPEG', 
                          'train/1040_cymbals/bing_cymbals_0053.jpg', \
                          'train/1040_cymbals/bing_cymbals_0099.jpg', 
                          'train/1040_cymbals/bing_cymbals_0214.jpg', 
                          'train/0062_table/n03201208_13855.JPEG',
                          'train/0062_table/n03201208_15935.JPEG',
                          'train/0062_table/n04379243_10278.JPEG',
                          'train/0062_table/n03201208_16993.JPEG',
                          'train/0062_table/n03201208_21690.JPEG',
                          'train/0062_table/n04379243_11621.JPEG',
                          'train/0062_table/n04379243_16384.JPEG',
                          'train/0062_table/bing_table_0078.jpg',
                          'train/0062_table/n02894337_1939.JPEG',
                          'train/0062_table/n03202354_1933.JPEG',
                          'train/0062_table/n04379243_15713.JPEG',
                          'train/0062_table/n04379243_372.JPEG',
                          'train/0062_table/n03063968_2178.JPEG',
                          'train/0062_table/n03201208_30581.JPEG',
                          'train/0062_table/n04379243_5120.JPEG',
                          'train/0062_table/n04379243_11852.JPEG',
                          'train/0062_table/n03201208_29391.JPEG',
                          'train/0062_table/n03201208_10515.JPEG',
                          'train/0062_table/n04603729_14500.JPEG',
                          'train/0062_table/n04379243_12323.JPEG',
                          'train/0062_table/n03201208_19794.JPEG',
                          'train/0062_table/n04379243_5858.JPEG',
                          'train/0062_table/flickr_table_0021.jpg',
                          'train/0062_table/n04379243_16234.JPEG',
                          'train/0062_table/n03063968_2169.JPEG',
                          'train/0062_table/n03063968_7903.JPEG',
                          'train/0921_bench/n04177820_22491.JPEG',
                          'train/0921_bench/n02828884_8338.JPEG',
                          'train/0921_bench/n03891251_2275.JPEG',
                          'train/0921_bench/n02828884_7404.JPEG', 
                          'train/0921_bench/n04177820_10903.JPEG',
                          'train/0921_bench/n04177820_11371.JPEG', 
                          'train/0921_bench/n04177820_3949.JPEG',
                          'train/1773_couch/n03115762_484.JPEG',
                          'train/1773_couch/n04256520_16621.JPEG',
                          'train/1773_couch/n03115762_17201.JPEG',
                          'train/1773_couch/n03115762_9819.JPEG',
                          'train/1773_couch/n04256520_15441.JPEG',
                          'train/1773_couch/n04256520_6081.JPEG',
                          'train/1773_couch/n04256520_23659.JPEG',
                          'train/1773_couch/n04256520_4917.JPEG',
                          'train/1773_couch/n04256520_2878.JPEG',
                          'train/1773_couch/n03115897_1054.JPEG',
                          'train/1773_couch/n03115762_8348.JPEG',
                          'train/1773_couch/n03115762_17720.JPEG',
                          'train/0768_television/n03072440_8730.JPEG',
                          'train/0768_television/n04404412_4214.JPEG',
                          'train/0768_television/n04404412_16931.JPEG',
                          'train/0768_television/n04404412_20014.JPEG',
                          'train/0768_television/n06277280_70993.JPEG',
                          'train/0768_television/n04404412_18229.JPEG',
                          'train/0768_television/n04405907_10460.JPEG',
                          'train/0768_television/n06278475_7779.JPEG',
                          'train/0768_television/n04404412_8650.JPEG',
                          'train/0768_television/n04404412_2545.JPEG',
                          'train/2137_chair/n03376595_7245.JPEG',
                          'train/2137_chair/n03002096_505.JPEG',
                          'train/2137_chair/n03376595_6545.JPEG',
                          'train/2137_chair/n03002096_9.JPEG',
                          'train/2137_chair/n03632729_2806.JPEG',
                          'train/2137_chair/n03632729_5625.JPEG',
                          'train/2137_chair/n04331277_823.JPEG',
                          'train/2137_chair/n03632729_1409.JPEG',
                          'train/2137_chair/n04099969_15627.JPEG',
                          'train/2137_chair/n04331277_6131.JPEG',
                          'train/2137_chair/n04099969_10364.JPEG',
                          'train/2137_chair/n03632729_464.JPEG',
                          'train/2137_chair/n03376595_3795.JPEG',
                          'train/2137_chair/n03376595_8642.JPEG',
                          'train/2137_chair/n04331277_6971.JPEG',
                          'train/2137_chair/n03376595_1104.JPEG',
                          'train/0182_refrigerator/n04070727_3395.JPEG', 
                          'train/0182_refrigerator/n03273913_15949.JPEG', 
                          'train/0182_refrigerator/n03170635_12179.JPEG',
                          'train/0182_refrigerator/n03170635_985.JPEG', 
                          'train/0182_refrigerator/n04070727_2366.JPEG', 
                          'train/0182_refrigerator/n04070727_4141.JPEG', 
                          'train/0182_refrigerator/n04070727_20748.JPEG',
                          'train/0182_refrigerator/n04070727_3627.JPEG',
                          'train/0182_refrigerator/n03273913_2934.JPEG', 
                          'train/0182_refrigerator/n04070727_22748.JPEG', 
                          'train/0182_refrigerator/n04070727_4033.JPEG',
                          'train/0182_refrigerator/n04070727_18024.JPEG',
                          'train/0182_refrigerator/n03170635_35893.JPEG', 
                          'train/0182_refrigerator/n04070727_24320.JPEG',
                          'train/0182_refrigerator/n03273913_4935.JPEG',
                          'train/0182_refrigerator/n04070727_3227.JPEG', 
                          'train/0182_refrigerator/n04070727_15272.JPEG',
                          'train/0182_refrigerator/n03170635_12835.JPEG',
                          'train/0182_refrigerator/n04070727_23605.JPEG',
                          'train/0182_refrigerator/n04070727_16159.JPEG',
                          'train/0182_refrigerator/n03273913_19950.JPEG', 
                          'train/0182_refrigerator/n04070727_29640.JPEG', 
                          'train/0182_refrigerator/n04070727_3982.JPEG',
                          'train/0250_lamp/n04380533_1195.JPEG',
                          'train/0250_lamp/n03636248_4731.JPEG',
                          'train/0250_lamp/n03636248_3351.JPEG',
                          'train/0250_lamp/n03636248_7153.JPEG',
                          'train/0077_ship/n04194289_11007.JPEG',
                          'train/0077_ship/n02965300_5205.JPEG',
                          'train/0077_ship/n04194289_8651.JPEG',
                          'train/0077_ship/n02965300_9812.JPEG', 
                          'train/0149_truck/n04467665_32472.JPEG',
                          'train/0149_truck/n03256166_8838.JPEG',
                          'train/0149_truck/n04467665_45502.JPEG',
                          'train/0009_car/n04285008_5056.JPEG',
                          'train/0009_car/n02960352_13935.JPEG',
                          'train/0009_car/n03268790_5880.JPEG',
                          'train/0009_car/n04285008_13645.JPEG',
                          'train/0089_bus/n04487081_11176.JPEG',
                          'train/0089_bus/n04212165_1234.JPEG', 
                          'train/0341_motorcycle/n03790512_21626.JPEG',
                          'train/0341_motorcycle/n03790512_2741.JPEG',
                          'train/0341_motorcycle/n03790512_5490.JPEG',
                          'train/0341_motorcycle/n03790512_3471.JPEG', 
                          'train/0341_motorcycle/n03790512_5640.JPEG',
                          'train/0341_motorcycle/n03790512_13080.JPEG',
                          'train/0341_motorcycle/n03790512_14043.JPEG', 
                          'train/0341_motorcycle/n03790512_8943.JPEG',
                          'train/0341_motorcycle/n04466871_7497.JPEG', 
                          'train/0341_motorcycle/n03790512_4502.JPEG', 
                          'train/0341_motorcycle/n10333838_16292.JPEG', 
                          'train/1112_canoe/n03254374_1572.JPEG',
                          'train/1112_canoe/n02951358_11388.JPEG',
                         ]

    return ecoset_ims_exclude

def get_trumpet_filenames():

    # this is for the trumpet category (called "bugle" in ecoset)
    # a lot of the images in ecoset for "bugle" are actually other brass 
    # instruments, so i am picking the actual trumpets manually here.

    good_names = ['train/1307_bugle/n02793089_7034.JPEG',
                 'train/1307_bugle/n03110669_13872.JPEG',
                 'train/1307_bugle/n03110669_29511.JPEG',
                 'train/1307_bugle/n03110669_39147.JPEG',
                 'train/1307_bugle/n03369276_5627.JPEG',
                 'train/1307_bugle/n02793089_18260.JPEG',
                 'train/1307_bugle/n02793089_16743.JPEG',
                 'train/1307_bugle/n03110669_60721.JPEG',
                 'train/1307_bugle/n03110669_4950.JPEG',
                 'train/1307_bugle/n04141198_4482.JPEG',
                 'train/1307_bugle/n02793089_5107.JPEG',
                 'train/1307_bugle/n02793089_9013.JPEG',
                 'train/1307_bugle/n03369276_4866.JPEG',
                 'train/1307_bugle/n02804252_2640.JPEG',
                 'train/1307_bugle/n02793089_780.JPEG',
                 'train/1307_bugle/n02804252_20619.JPEG',
                 'train/1307_bugle/n03110669_4467.JPEG',
                 'train/1307_bugle/n03110669_12321.JPEG',
                 'train/1307_bugle/n03110669_129909.JPEG',
                 'train/1307_bugle/n03369276_8508.JPEG',
                 'train/1307_bugle/n02912894_6075.JPEG',
                 'train/1307_bugle/n02912894_19418.JPEG',
                 'train/1307_bugle/n02804252_8579.JPEG',
                 'train/1307_bugle/n02793089_13919.JPEG',
                 'train/1307_bugle/n03110669_63279.JPEG',
                 'train/1307_bugle/n03110669_11562.JPEG',
                 'train/1307_bugle/n03110669_26581.JPEG',
                 'train/1307_bugle/n02793089_11325.JPEG',
                 'train/1307_bugle/n04141198_2667.JPEG',
                 'train/1307_bugle/n03110669_20521.JPEG',
                 'train/1307_bugle/n02793089_6196.JPEG',
                 'train/1307_bugle/n02793089_20875.JPEG',
                 'train/1307_bugle/n03110669_6328.JPEG',
                 'train/1307_bugle/n03110669_29251.JPEG',
                 'train/1307_bugle/n03110669_27808.JPEG',
                 'train/1307_bugle/n03110669_2656.JPEG', 
                 'train/1307_bugle/n03369276_6164.JPEG', 
                 'train/1307_bugle/n03110669_21012.JPEG', 
                 ]
    

    return good_names