

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import time

from PIL import Image

from google.colab import files
uploaded = files.upload()

img = Image.open('lion.jpg')
imggray = img.convert('LA')
plt.figure(figsize=(9, 6))
plt.imshow(imggray);

imgmat = np.array(list(imggray.getdata(band=0)), float)
imgmat.shape = (imggray.size[1], imggray.size[0])
imgmat = np.matrix(imgmat)
plt.figure(figsize=(9,6))
plt.imshow(imgmat, cmap='gray');

imgmat.shape

U, sigma, V = np.linalg.svd(imgmat)

P, D, Q = np.linalg.svd(imgmat, full_matrices=False)
X_a = P @ np.diag(D) @ Q
print(np.std(imgmat), np.std(X_a), np.std(imgmat - X_a))
print('Is X close to X_a?', np.isclose(imgmat, X_a).all())
plt.imshow(X_a, cmap='gray');

for i in range(10, 51, 10):
    reconstimg = np.matrix(U[:, :i]) * np.diag(sigma[:i]) * np.matrix(V[:i, :])
    plt.imshow(reconstimg, cmap='gray')
    title = "n = %s" % i
    plt.title(title)
    plt.show()
