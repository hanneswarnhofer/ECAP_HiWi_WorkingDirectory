from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.torch.base import BaseModel


class StereoCNN(BaseModel):
    def __init__(self, tasks):
        super().__init__(tasks)
        self.tasks = tasks
        self.output_names = tasks
        self.conv1 = nn.Conv2d(1, 8, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(8, 16, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(16, 32, 3, 1, padding="same")
        self.conv4 = nn.Conv2d(32, 64, 3, 1, padding="same")
        self.conv5 = nn.Conv2d(64, 128, 3, 1, padding="same")
        self.conv6 = nn.Conv2d(128, 256, 3, 1, padding="same")

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(1024, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc_primary = nn.Linear(100, 2)
        self.fc_energy = nn.Linear(100, 1)
        self.fc_axis = nn.Linear(100, 3)

    def forward(self, inp={"ct1": None, "ct2": None, "ct3": None, "ct4": None, "ct5": None}):
        out = []
        for x in [inp["ct1"], inp["ct2"], inp["ct3"], inp["ct4"]]:
            x = self.conv1(x)
            x = F.elu(x)
            x = self.conv2(x)
            x = F.elu(x)
            x = F.max_pool2d(x, 2)
            x = self.conv3(x)
            x = F.elu(x)
            x = self.conv4(x)
            x = F.elu(x)
            x = F.max_pool2d(x, 2)
            x = self.conv5(x)
            x = F.elu(x)
            x = self.conv6(x)
            x = F.elu(x)
            x = F.max_pool2d(x, kernel_size=x.size()[2:])
            x = torch.flatten(x, 1)
            out.append(x)

        # merge output of telescope images
        z = torch.concat(out, axis=-1)
        z = self.dropout1(z)
        z = self.fc1(z)
        z = F.elu(z)
        z = self.dropout2(z)
        z = self.fc2(z)
        z = F.elu(z)
        z = self.dropout3(z)

        outputs = {}
        if "primary" in self.tasks:
            outputs["primary"] = F.softmax(self.fc_primary(z), dim=-1)

        if "energy" in self.tasks:
            outputs["energy"] = self.fc_energy(z)

        if "axis" in self.tasks:
            outputs["axis"] = self.fc_axis(z)

        return outputs


class DummyCNN(BaseModel):
    def __init__(self, tasks):
        super().__init__(tasks)
        self.tasks = tasks
        self.output_names = tasks
        self.conv1 = nn.Conv2d(4, 8, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(8, 16, 3, 1, padding="same")
        self.conv3 = nn.Conv2d(16, 32, 3, 1, padding="same")
        self.conv4 = nn.Conv2d(32, 64, 3, 1, padding="same")
        self.conv5 = nn.Conv2d(64, 128, 3, 1, padding="same")
        self.dropout1 = nn.Dropout(0.5)

        self.conv1_ct5 = nn.Conv2d(1, 8, 3, 1, padding="same")
        self.conv2_ct5 = nn.Conv2d(8, 16, 3, 1, padding="same")
        self.conv3_ct5 = nn.Conv2d(16, 32, 3, 1, padding="same")
        self.conv4_ct5 = nn.Conv2d(32, 64, 3, 1, padding="same")
        self.conv5_ct5 = nn.Conv2d(64, 128, 3, 1, padding="same")
        self.dropout1_ct5 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(256, 2)
        self.fc2 = nn.Linear(256, 1)
        self.fc3 = nn.Linear(256, 3)

    def forward(self, inp={"ct1": None, "ct2": None, "ct3": None, "ct4": None, "ct5": None}):
        x = torch.concat([inp["ct1"], inp["ct2"], inp["ct3"], inp["ct4"]], axis=1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = torch.flatten(x, 1)

        y = self.conv1_ct5(inp["ct5"])
        y = F.relu(y)
        y = self.conv2_ct5(y)
        y = F.relu(y)
        y = F.max_pool2d(y, 2)
        y = self.conv3_ct5(y)
        y = F.relu(y)
        y = self.conv4_ct5(y)
        y = F.relu(y)
        y = F.max_pool2d(y, 2)
        y = self.conv5_ct5(y)
        y = F.relu(y)
        y = F.max_pool2d(y, kernel_size=y.size()[2:])
        y = torch.flatten(x, 1)
        x = self.dropout1(x)

        outputs = {}
        if "primary" in self.tasks:
            z = torch.concat([x, y], axis=-1)
            outputs["primary"] = F.softmax(self.fc1(z), dim=-1)

        if "energy" in self.tasks:
            z = torch.concat([x, y], axis=-1)
            outputs["energy"] = self.fc2(z)

        if "axis" in self.tasks:
            z = torch.concat([x, y], axis=-1)
            outputs["axis"] = self.fc3(z)

        return outputs
