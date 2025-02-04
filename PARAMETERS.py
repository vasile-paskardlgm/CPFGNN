
cora = {
    'lr':0.001,
    'prop_g_lr':0.01,
    'weight_decay':0.0005,
    'prop_g_wd':0,
    'dropout':0.5,
    'dprate':0,
    'prop_f_lr':0.005,
    'prop_f_wd':0,
    'r_train':0.6,
    'r_val':0.2
}

citeseer = {
    'lr':0.01,
    'prop_g_lr':0.01,
    'weight_decay':0,
    'prop_g_wd':0,
    'dropout':0.5,
    'dprate':0.5,
    'prop_f_lr':0.05,
    'prop_f_wd':0.0005,
    'r_train':0.6,
    'r_val':0.2
}

pubmed = {
    'lr':0.05,
    'prop_g_lr':0.005,
    'weight_decay':0.0005,
    'prop_g_wd':0,
    'dropout':0,
    'dprate':0,
    'prop_f_lr':0.05,
    'prop_f_wd':0.05,
    'r_train':0.6,
    'r_val':0.2
}

roman_empire = {
    'lr':0.05,
    'prop_g_lr':0.1,
    'weight_decay':0.0005,
    'prop_g_wd':0,
    'dropout':0,
    'dprate':0,
    'prop_f_lr':0.01,
    'prop_f_wd':0.0005,
    'r_train':0.5,
    'r_val':0.25
}

amazon_ratings = {
    'lr':0.05,
    'prop_g_lr':0.1,
    'weight_decay':0,
    'prop_g_wd':0,
    'dropout':0,
    'dprate':0.5,
    'prop_f_lr':0.01,
    'prop_f_wd':0.0005,
    'r_train':0.5,
    'r_val':0.25
}

######################
PARAMS = {
    'cora':cora,
    'citeseer':citeseer,
    'pubmed':pubmed,
    'amazon-ratings':amazon_ratings,
    'roman-empire':roman_empire
}