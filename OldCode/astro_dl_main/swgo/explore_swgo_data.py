from astropy import units as u
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from tools.utils import config

CONFIG = config()

tree = ET.parse('/home/atuin/b129dc/b129dc11/share/swgo/survey_ref_config.xml')
root = tree.getroot()

all_tanks = list()
for tank in root.iter('tank'):
    all_tanks.append([int(tank.attrib['id']),
                      float(tank[2][0].text) * u.cm.to('m'),
                      float(tank[2][1].text) * u.cm.to('m'),
                      ])
all_tanks = np.array(all_tanks)


all_tanks = pd.DataFrame(all_tanks, columns=['id', 'x', 'y'])
all_tanks['id'] = all_tanks['id'].astype('int32')
all_tanks[['x', 'y']] = all_tanks[['x', 'y']].round(2)

all_channels = list()
for tank in root.iter('tank'):
    x_tank = float(tank[2][0].text) * u.cm.to('m')
    y_tank = float(tank[2][1].text) * u.cm.to('m')
    z_tank = float(tank[2][2].text) * u.cm.to('m')

    for channel in tank.iter('channels'):

        for i in range(2):  # number of channels
            x_channel = float(channel[i][3][0].text) * u.cm.to('m')
            y_channel = float(channel[i][3][1].text) * u.cm.to('m')
            z_channel = float(channel[i][3][2].text) * u.cm.to('m')
            all_channels.append([int(channel[i].attrib['id']),
                                 x_tank + x_channel,
                                 y_tank + y_channel,
                                 z_tank + z_channel])

all_channels = np.array(all_channels)

all_channels = pd.DataFrame(all_channels, columns=['id', 'x', 'y', 'z'])
all_channels['id'] = all_channels['id'].astype('int32')
all_channels[['x', 'y']] = all_channels[['x', 'y']].round(1)
all_channels['z'] = all_channels['z'].round(2)

all_channels.head(5)
cut_UC = "z == 0.06"
cut_LC = "z == 0.07"

fig, ax = plt.subplots(figsize=(8, 8))
upper_cells = all_channels.query(cut_UC)
lower_cells = all_channels.query(cut_LC)
plt.scatter(lower_cells['x'], lower_cells['y'], 1.5)
plt.xlabel('X [m]')
plt.ylabel('Y [m]')

fig.savefig(CONFIG.log_dir + "/swgo_layout.png")
