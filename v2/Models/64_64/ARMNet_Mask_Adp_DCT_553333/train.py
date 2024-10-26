from interface import Interface

if __name__=='__main__':
    Interface().train(epoch=200, learning_rate=8e-4, batch_size=25, save_freq=1, noise_level=None, train_mask=True)
