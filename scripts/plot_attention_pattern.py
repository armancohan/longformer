def main():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams['axes.linewidth'] = 5.1
    import numpy as np
    from matplotlib.colors import ListedColormap
    from matplotlib import colors


    def kth_diag_indices(a, k):
        rows, cols = np.diag_indices_from(a)
        if k < 0:
            return rows[-k:], cols[:k]
        elif k > 0:
            return rows[:-k], cols[k:]
        else:
            return rows, cols

    nrows, ncols = 24, 24
    image = np.zeros(nrows*ncols)

    image = image.reshape((nrows, ncols))

    for i in range(-3,4):
        image[kth_diag_indices(image, i)] = 40

    for i in [0,1, 5, 15]:
        image[i, :] = 40
        image[:, i] = 40

    image[kth_diag_indices(image, 0)] = 70

    image1 = np.ones(nrows*ncols) * 50
    image1 = image1.reshape((nrows, ncols))
    image1[kth_diag_indices(image, 0)] = 70

    image2 = np.zeros(nrows*ncols)
    image2 = image2.reshape((nrows, ncols))
    for i in range(-5,6):
        image2[kth_diag_indices(image2, i)] = 40
    image2[kth_diag_indices(image, 0)] = 70

    image3 = np.zeros(nrows*ncols)
    image3 = image3.reshape((nrows, ncols))
    for i in range(-7,8,2):
        image3[kth_diag_indices(image3, i)] = 40
    image3[kth_diag_indices(image, 0)] = 70


    ## Sliding chunks
    image4 = np.zeros((nrows, ncols)) + 50
    for i in range(-5,6):
        image4[kth_diag_indices(image2, i)] = 80
    
    # block_size = 8
    # for i in range(nrows//block_size):
    #     image4[i * block_size: (i+1) * block_size, i * block_size: (i+1) * block_size] = 80
    #     image4[(i * block_size) + block_size // 2: (i+1) * block_size + block_size // 2, 
    #             i * block_size + block_size // 2 : (i+1) * block_size + block_size // 2] = 80

    # masking 
    # for j in range(block_size // 2, block_size):
    #     for k in range(0, block_size // 2):
    #         if j - k  > block_size // 2:
    #             for i in range(nrows//block_size):
    #                 image4[i * block_size + j, i * block_size + k] = 30
    #                 image4[i * block_size + k, i * block_size + j] = 30
    #                 try:
    #                     image4[i * block_size + j + block_size // 2, i * block_size + k + block_size // 2] = 30
    #                     image4[i * block_size + k + block_size // 2, i * block_size + j + block_size // 2] = 30
    #                 except IndexError:
    #                     pass
    image4[kth_diag_indices(image, 0)] = 80



    row_labels = range(nrows)
    col_labels = [rf'$w_{{{i}}}$' for i in range(nrows)]
    # cmap = colors.ListedColormap(['white', '', 'blue'])
    # bounds = [0,10,20]
    # norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()

    # ax.imshow(image4, cmap=plt.get_cmap('BuGn'), vmin=0, vmax=100)
    ax.imshow(image4, cmap=plt.get_cmap('BrBG'), vmin=0, vmax=100)
# draw gridlines
    ax.grid(which='minor', axis='both', linestyle='-', color='k', linewidth=0.05)
    # plt.matshow(image, cmap='Purples', alpha=0.6)
    col_labels = [rf'$w_{{{i}}}$' for i in range(nrows)]
    ax.set_xticks(np.arange(ncols) - 0.5, col_labels)
    ax.set_yticks(np.arange(nrows) - 0.5, col_labels)
    # plt.xticks(range(ncols), col_labels, rotation=90)
    # plt.yticks(range(nrows), col_labels)
    plt.tick_params(axis='both', left='off', top='off', right='off', bottom='off', labelleft='off', labeltop='off', labelright='off', labelbottom='off')
    plt.show()
    # plt.savefig('fig-attn-window1.pdf', bbox_inches='tight')

if __name__ == "__main__":
    main()
