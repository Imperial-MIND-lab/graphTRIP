"""
Utility functions for file management.

License: BSD-3-Clause
Author: Hanna M. Tolle
Date: 2024-10-31
"""

import os
import shutil


def project_root():
    '''Returns project root path (where utils/ is located).'''
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def add_project_root(dir_name=None):
    '''Returns dir_name if it's already rooted at project_root(), otherwise joins with project_root().'''
    if dir_name is None:
        return None
    if dir_name.startswith(project_root()):
        return dir_name
    return os.path.join(project_root(), dir_name)

def rm_project_root(dir_name=None):
    '''Removes everything in the file path before /graphTRP/ (inclusive).'''
    if dir_name is None:
        return None
    # Find index of graphTRIP in path and remove everything before it (inclusive)
    idx = dir_name.find('/graphTRIP/')
    if idx == -1:
        return dir_name
    return dir_name[idx + len('/graphTRIP/'):]

def raw_data_dir():
    '''Returns file path to raw (non-parcellated) neuroimaging data.'''
    usr = os.getenv('USER')
    if usr=='hanna':
        return os.path.expanduser("~/Public/neuroimaging-data")
    elif usr=='hmt23':
        return os.path.expanduser("~/data")
    else:
        raise ValueError("Unknown username")
 

def get_raw_filename(study='psilodep2', session='before'):
    if study=='psilodep2' or study=='psilodep1':
        return f'{session}_rest_rdsmffms6FWHM_bd_M_V_DV_WMlocal2_modecorr.nii.gz'
    else:
        raise ValueError("Unknown study.")


def get_subject_id(i, prefix='S'):
    '''Returns zero-padded subject ID string.'''
    return prefix + '0' + str(i+1) if i<9 else prefix + str(i+1)
    
    
def get_filepath(root=None, study='psilodep2', session=None, atlas=None, subject=None):
    '''
    Returns data file path string. This function should be used to enforce consistent file paths.
    The max full file path is: project_root/data/raw/study/session/atlas/subject/
    '''
    # Define root file path
    if root == None:
        root = os.path.join(project_root(), 'data')
    filepath = os.path.join(root, 'raw', study)

    # Dataset with multiple sessions
    if session:
        filepath = os.path.join(filepath, session)

    # Parcellated versus non-parcellated data
    if atlas:
        filepath = os.path.join(filepath, atlas)

        # Subject folder
        if not subject==None:
            filepath = os.path.join(filepath, get_subject_id(subject))
    
    return filepath

def move_files_into_parentdir(parent_dir, sub_dir):
    '''
    Copies all unique files from a subdirectory into the parent directory.
    '''
    for root, dirs, files in os.walk(sub_dir):
        for file in files:
            src_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(parent_dir, file)
            if not os.path.exists(dest_file_path):
                shutil.copy2(src_file_path, dest_file_path)

def files_missing(dir, required_files):
    '''
    Checks if all required files are present in the directory.
    '''
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(dir, f))]
    if missing_files:
        return True
    else:
        return False
    
def remove_dir(dir):
    '''
    Removes a directory and its contents.
    '''
    if os.path.exists(dir):
        shutil.rmtree(dir)
