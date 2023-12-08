#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 14:11:48 2018

@author: mase
"""

# https://github.com/keras-team/keras/issues/6499

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras.utils import Sequence
import glob, os, numpy as np

#example:
#train_dir='/data/workspace/mseeland/DMVCNN/datasets/plantclef2016/test/'
#views=['flower', 'leaf','flower']
#generator = ImageDataGenerator()
#vgen0 = generator.flow_from_directory(os.path.join(train_dir, views[0]))


        
def clean_dataset(set_dir):
    idx = 0
    for idx,bad_file in enumerate(glob.glob(os.path.join(set_dir, '*', '.*')), start=1):
        os.remove(bad_file)
    if idx:
        print('{} invalid files removed'.format(idx))

class MultiGenerator(Sequence):
    
    def __init__(self, view_generators, shuffle):
        self.view_generators = view_generators
        self.shuffle = shuffle
        self.batch_size = view_generators[0].batch_size
        self.num_steps = len(view_generators[0])
        self.indices = np.arange(view_generators[0].samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
        
    def __len__(self):
        return self.num_steps
    
    def __getitem__(self, idx):
#        print('\t {}'.format(idx)) # ensure correct ordering when using multiple workers!
        inds = self.indices[idx * self.batch_size:(idx+1) * self.batch_size]
        data = []
        for gen in self.view_generators:
            minibatch = gen._get_batches_of_transformed_samples(inds)
            if gen.ismasked:
                data.append(np.multiply(minibatch[0],0))
            else:
                data.append(minibatch[0])
        labels = minibatch[1]
        return data, labels
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def check_labels(view_generators):
    while True:
        gen_i = []
        for gen in view_generators:
            gen_i.append(gen.next())
        yield [y.argmax() for y in [x[1] for x in gen_i]]

        

def data_iterator(set_dir, input_size, batch_size, views, train=False, 
                  seed=0, masked_views=[], **kwargs):
    
    for view in views:
        clean_dataset(os.path.join(set_dir, view))
    
    generator = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            **kwargs
            )
    
    if train:
        save_dir = None#'/data/workspace/mseeland/DMVCNN/testdata_temp'
    else:
        save_dir = None#'/data/workspace/mseeland/DMVCNN/testdata_temp'
    
    view_generators = []
    for idx,view in enumerate(views):
        # one generator per view        
        print('{} view "{}"'.format('train' if train else 'test',view))
        view_generators.append(
                generator.flow_from_directory(
                        os.path.join(set_dir, view),
                        target_size=input_size, 
                        class_mode='categorical', 
                        shuffle=False, #train, # shuffle only for training data
                        batch_size=batch_size,
                        seed=seed,
                        save_to_dir=save_dir,
                        save_format='jpeg',
                        save_prefix='train_{}_{}'.format(view,idx) if train else 'test_{}_{}'.format(view,idx)
                        )
                )
        if view in masked_views:
            view_generators[-1].ismasked = True
        else:
            view_generators[-1].ismasked = False

    num_steps = len(view_generators[0])
    print('steps: {}'.format(num_steps))
    
    print('\n')
                
    batches_generator = MultiGenerator(view_generators, shuffle=train)

    return batches_generator, num_steps, get_class_weights(view_generators[0]), view_generators[0].class_indices.keys(), view_generators[0].filenames, view_generators[0].classes

def get_class_weights(generator):
    counts = np.bincount(generator.classes)
    counts = max(counts)/counts
    class_weights = dict()
    for c in zip(generator.class_indices.values(), counts):
        class_weights[c[0]] = c[1]
    return class_weights