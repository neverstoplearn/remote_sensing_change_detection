import numpy as np
import matplotlib.pyplot as plt

class Encode_Decode():
    
    
    def get_pascal_labels(self):
        
        """Load the mapping that associates pascal classes with label colors

        Returns:
            np.ndarray with dimensions (21, 3)
        """
        return np.array([[  0,   0,   0],
                         [0, 200 ,0],
                       [150, 250,   0],
                       [150, 200, 150],
                       [200,   0, 200],
                       [150,   0, 250],
                       [150, 150, 250],
                       [250, 200,   0],
                       [200, 200,   0],
                       [200,   0,   0],
                       [250,   0, 150],
                       [200, 150, 150],
                       [250, 150, 150],
                       [  0,   0, 200],
                       [  0, 150, 200],
                       [  0, 200, 250]])
    
    
    def encode_segmap(self, mask):
        """Encode segmentation label images as pascal classes

        Args:
            mask (np.ndarray): raw segmentation label image of dimension
              (M, N, 3), in which the Pascal classes are encoded as colours.

        Returns:
            (np.ndarray): class map with dimensions (M,N), where the value at
            a given location is the integer denoting the class index.
        """
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for ii, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
        label_mask = label_mask.astype(int)
        return label_mask



    def decode_segmap(self,label_mask, plot=False):
        """Decode segmentation class labels into a color image

        Args:
            label_mask (np.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.

        Returns:
            (np.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_pascal_labels()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, len(label_colours)):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b 
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb.astype(int)