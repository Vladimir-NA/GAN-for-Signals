import torch
from torch import nn
import math
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
import os
import numpy as np

project_dir_path = os.getcwd() + "\\"
graphs_dp = project_dir_path + "3DGraphs\\"
train_data_dp = project_dir_path + "train_data\\"
generated_files_dp = project_dir_path + "generated_files\\"

torch.manual_seed(111)

def read_file(fn):
    f1 = open(fn, 'r')
    data = []
    for line in f1:
        data.append([float(x) for x in line.split()])
    f1.close()
    return data

def xyz(data):
    x = []
    y = []
    z = []
    for v in data:
        x.append(v[0])
        y.append(v[1])
        z.append(v[2])
    return x, y, z

def make_fig(x, y, z, color, name ='noname'):

    fig = go.Scatter3d( x=x,
                        y=y,
                        z=z,
                        marker=dict(opacity=0.9,
                                    reversescale=True,
                                    color=color,
                                    colorscale='blues',
                                    size=5),
                        line=dict(width=0.02),
                        mode='markers',
                        name=name)
    return fig

def get_graph(figures, fname):
    mylayout = go.Layout(scene=dict(xaxis=dict(title="X"),
                                    yaxis=dict(title="Y"),
                                    zaxis=dict(title="Z")), )
    plotly.offline.plot({"data": figures,
                         "layout": mylayout},
                        auto_open=False,
                        filename=(fname))

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid())

    def forward(self, x):
        output = self.model(x)
        return output

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 3))

    def forward(self, x):
        output = self.model(x)
        return output

generator = Generator()
discriminator = Discriminator()

def create_loss_graph(epochs_num, gen_losses, dis_losses):
    E = np.arange(0, epochs_num)
    plt.plot(E, gen_losses, color='r', label='Generator loss')
    plt.plot(E, dis_losses, color='g', label='Discriminator loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    f = 1;



def train(all_data, train_loader, batch_size, batch_q, epochs = 10000, lr = 0.001, minD = 0.0001, enable_stop = True):
    loss_function = nn.BCELoss()
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)
    gen_losses = []
    dis_losses = []
    for epoch in range(epochs):
        for n, (real_samples, _) in enumerate(train_loader):
            
            real_samples_labels = torch.ones((batch_size, 1))
            latent_space_samples = torch.randn((batch_size, 3))
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((batch_size, 1))
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat(
                (real_samples_labels, generated_samples_labels))

            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(
                output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()


            latent_space_samples = torch.randn((batch_size, 3))

            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(
                output_discriminator_generated, real_samples_labels)
            loss_generator.backward()
            optimizer_generator.step()
        gen_losses.append(loss_generator.item())
        dis_losses.append(loss_discriminator.item())
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")

        if epoch % 100 == 0:
            generate_signals(all_data[0], batch_size, graphName='epoch' + str(epoch) + '.html')

        d = max(loss_discriminator.item(), loss_generator.item()) - min(loss_discriminator.item(), loss_generator.item())
        if (d <= minD) & (enable_stop):
            print("stopped" )
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")
            create_loss_graph(epoch + 1, gen_losses, dis_losses)
            break;

def generate_signals(data, signal_len, graphName ='test1'):
    latent_space_samples = torch.randn(signal_len, 3)
    generated_samples = generator(latent_space_samples)
    generated_samples = generated_samples.detach()
    x, y, z = xyz(data)
    fig1 = make_fig(x, y, z, 'green', name='Real')
    genX = generated_samples[:, 0].tolist()
    genY = generated_samples[:, 1].tolist()
    genZ = generated_samples[:, 2].tolist()
    fig2 = make_fig(genX, genY, genZ, 'firebrick', name='Fake')
    get_graph([fig1, fig2], graphs_dp + graphName)
    gen_data = data_from_xyz(genX, genY, genZ)
    return gen_data



def get_all_data(data_dir_path):
    all_data = []
    for fn in os.listdir(data_dir_path):
        file_path = data_dir_path + "\\" + fn
        data = read_file(file_path)
        all_data.append(data)
    return all_data

def create_data_loader(all_data, batch_size):
    all_x = []
    all_y = []
    all_z = []
    data_len = 0
    for data in all_data:
        x, y, z = xyz(data)
        all_x += x
        all_y += y
        all_z += z
        data_len += len(data)
    train_data = torch.zeros((data_len, 3))
    train_data[:, 0] = torch.tensor(all_x)
    train_data[:, 1] = torch.tensor(all_y)
    train_data[:, 2] = torch.tensor(all_z)
    train_labels = torch.zeros(data_len)
    train_set = [(train_data[i], train_labels[i]) for i in range(data_len)]
    data_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return data_loader

def data_from_xyz(X,Y,Z):
    data_len = len(X)
    data = []
    for i in range(0, data_len):
        v = []
        v.append(X[i])
        v.append(Y[i])
        v.append(Z[i])
        data.append(v)
    return data

def write_file(fn, data):
    file = open(fn, 'w')
    for signal in data:
        line = str(round(signal[0], 3)) + " " + str(round(signal[1], 3)) + " " + str(round(signal[2], 3))
        file.write(line + '\n')
    file.close()


def testGAN(data_dir_path, gen_file_name = "none"):
    all_data = get_all_data(data_dir_path)
    batch_size = len(all_data[0])
    bath_q = len(all_data)
    data_loader = create_data_loader(all_data, batch_size)
    num_epochs = 5000
    lr = 0.001
    minD = 0.001
    train(all_data, data_loader, batch_size, bath_q, epochs=num_epochs, lr=lr, minD=minD, enable_stop=True)
    gen_data = generate_signals(all_data[0], batch_size, graphName='3DGraph1.html')
    write_file(gen_file_name, gen_data)

сlimb_stairs_dp = train_data_dp + "climb_stairs"
getup_bed_dp = train_data_dp + "getup_bed"
walk_dp = train_data_dp + "walk"

gen_file_name = "getup_bed"

testGAN(getup_bed_dp, gen_file_name = generated_files_dp + gen_file_name + ".txt")



