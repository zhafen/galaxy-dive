#!/usr/bin/env python
'''Script to make movies using ffmpeg

@author: Zach Hafen
@contact: zachary.h.hafen@gmail.com
@status: Development
'''

import os
import shutil
import sys

import galaxy_dive.plot_data.plotting as gen_plot

########################################################################
########################################################################

movie_dir_rel_path = sys.argv[1]

if len( sys.argv ) > 2:
  movie_quality = int( sys.argv[2] )
else:
  movie_quality = 1

movie_dir = os.path.abspath( movie_dir_rel_path )

dirs_to_use = os.listdir( movie_dir )

for dir_to_use in dirs_to_use:

  if os.path.isdir( dir_to_use ):

    img_dir = os.path.join( movie_dir, dir_to_use )

    file_pattern = '{}_*.png'.format( dir_to_use )

    movie_save_file = os.path.join( movie_dir, '{}.mp4'.format( dir_to_use ) )

    gen_plot.make_movie( img_dir, file_pattern, movie_save_file, movie_quality )
