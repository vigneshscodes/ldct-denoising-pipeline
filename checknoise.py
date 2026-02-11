import pydicom
import matplotlib.pyplot as plt
ndct_path = r"D:\CT_Datasets\NDCT\manifest-1770741989405\LIDC-IDRI\LIDC-IDRI-0001\01-01-2000-NA-NA-30178\3000566.000000-NA-03192\1-005.dcm"
ldct_path = r"D:\CT_Datasets\LDCT\manifest-1770741989405\LIDC-IDRI\LIDC-IDRI-0001\01-01-2000-NA-NA-30178\3000566.000000-NA-03192\1-005.dcm"
ndct = pydicom.dcmread(ndct_path).pixel_array
ldct = pydicom.dcmread(ldct_path).pixel_array
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("NDCT")
plt.imshow(ndct, cmap="gray")
plt.subplot(1,2,2)
plt.title("LDCT")
plt.imshow(ldct, cmap="gray")
plt.show()
