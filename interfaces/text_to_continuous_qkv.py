import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os, sys, io
import librosa, librosa.display
import soundfile as sf
import torch
import pickle
import urllib.request

from tqdm import tqdm
from stqdm import stqdm
import json

import sys
sys.path.insert(0, '../')
from audioldm2 import text_to_audio, build_model, seed_everything, make_batch_for_text_to_audio
from audioldm2.latent_diffusion.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
    noise_like,
    extract_into_tensor,
)
from audioldm2.latent_diffusion.models.ddim import DDIMSampler
from audioldm2.utilities import *
from audioldm2.utilities.audio import *
from audioldm2.utilities.data import *
from audioldm2.utils import default_audioldm_config

from audioldm2.gaverutils import gaver_sounds

from diffusion_helper_qkv import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

model_name = 'audioldm_16k_crossattn_t5'
latent_t_per_second=25.6
sample_rate=16000
duration = 10.0 #Duration is minimum 10 secs. The generated sounds are weird for <10secs
guidance_scale = 3
random_seed = 42
n_candidates = 1
batch_size = 1
ddim_steps = 20

latent_diffusion = None


def get_config(filepath='config/config.json'):
    config = {}
    with open(filepath, 'r') as f:
        config = json.load(f)
    return config


def populate_prompts():
    prompts_map = {}
    prompts = get_config()
    for prompt in prompts:
        prompts_map[prompt['source_prompt']] = prompt        
    return prompts_map

@st.cache_resource
def get_model():
    print('Loading model')
    
    latent_diffusion = build_model(model_name=model_name)
    latent_diffusion.latent_t_size = int(duration * latent_t_per_second)

    print('Model loaded')
    return latent_diffusion


def generate_sample(latent_diffusion, source_text, target_text, target_selected_word_list, target_value_list, random_seed, ddim_steps):
    with st.spinner('Running...'):
        wav, img = sample_diffusion_attention_edit(latent_diffusion, source_text=source_text, target_text=target_text, \
                                                        target_selected_word_list=target_selected_word_list, target_value_list=target_value_list,\
                                                        random_seed=random_seed, ddim_steps=ddim_steps)
        
    return wav, img
   





def default_target_prompt():
    if 'prompt_noun_replacements' in st.session_state:
        st.session_state['prompt_noun_replacements'] = '-None-'

def main():
    
    css="""
    <style>
        [data-testid="stAppViewBlockContainer"] {
            padding: 2rem;
        }
    </style>
    """
    st.write(css, unsafe_allow_html=True)

    print('before get model')
    latent_diffusion = get_model()
    print('after get model')

    prompts_map = populate_prompts()


    st.markdown("<h3 style='text-align: center;'>Semantic Word Weighting w/ Text</h3>", unsafe_allow_html=True)

    prompt_selected =  st.selectbox('Select a prompt', prompts_map.keys(), key='prompt_selected', on_change=default_target_prompt)
    slider_words = prompts_map[prompt_selected]['source_slider_words']
    prompt_id = str(prompts_map[prompt_selected]['id'])

    prompt_seed = prompts_map[prompt_selected]['prompt_seed']

    num_ddim_steps = prompts_map[prompt_selected]['num_steps']

    st.session_state['selected_prompt_id'] = str(prompt_id)

    selected_word_list = []
    selected_word_value_list = []
    for slider_word in slider_words:
        selected_word_list.append(slider_word['word'])
        if 'slider_'+st.session_state['selected_prompt_id']+'_'+slider_word['word'] in st.session_state:
            selected_word_value_list.append(st.session_state['slider_'+st.session_state['selected_prompt_id']+'_'+slider_word['word']])
        else:
            selected_word_value_list.append(1)

    if st.session_state['prompt_selected'] != '-None-':
        s_wav, s_spec = generate_sample(latent_diffusion, source_text=None, target_text=prompt_selected, \
                                                            target_selected_word_list=selected_word_list, target_value_list=selected_word_value_list,\
                                                            random_seed=prompt_seed, ddim_steps=num_ddim_steps)

    display_text = prompt_selected
    for slider_word in slider_words:
        display_text = display_text.replace(slider_word['word'], "<span style='background-color: yellow; color:black;'>"+slider_word['word']+"</span>")
    st.markdown("<div style='text-align: center;'><h3>"+display_text+"</h3></div>", unsafe_allow_html=True)
    st.markdown("<br/>", unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns((0.2,0.1,0.4,0.05,0.25))
    
    with col1:
        st.markdown("<br/>", unsafe_allow_html=True)
        display_text = prompt_selected
        for slider_word in slider_words:
            display_text = display_text.replace(slider_word['word'], "<span style='background-color: yellow; color:black;'>"+slider_word['word']+"</span>")
        st.markdown("<div style='text-align: left;'>"+display_text+"</div>", unsafe_allow_html=True)
        st.markdown("<br/>", unsafe_allow_html=True)
        for slider_word in slider_words:
            slider_position=st.slider(slider_word['word'], min_value=slider_word['slider_range'][0], max_value=slider_word['slider_range'][1], \
                                      value=1.0, step=0.1,  format=None, key='slider_'+st.session_state['selected_prompt_id']+'_'+slider_word['word'], disabled=False)
    with col2:
        vert_space = '<div style="padding: 25%;">&nbsp;</div>'
        st.markdown(vert_space, unsafe_allow_html=True)
    with col3:
        st.image(s_spec)
        st.audio(s_wav, format="audio/wav", start_time=0, sample_rate=16000)

    st.markdown('<div style="text-align:center;color:white"><i>All audio samples on this page are generated with a sampling rate of 16kHz.</i></div>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()