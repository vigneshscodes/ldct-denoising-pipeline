import pydicom
import matplotlib.pyplot as plt
ldct_path = r"D:\CT_Datasets\LDCT\manifest-1770741989405\LIDC-IDRI\LIDC-IDRI-0001\01-01-2000-NA-NA-30178\3000566.000000-NA-03192\1-001.dcm"
proc_path = r"D:\CT_Datasets\LDCT_Processed\manifest-1770741989405\LIDC-IDRI\LIDC-IDRI-0001\01-01-2000-NA-NA-30178\3000566.000000-NA-03192\1-001.dcm"
ldct = pydicom.dcmread(ldct_path).pixel_array
proc = pydicom.dcmread(proc_path).pixel_array
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("LDCT")
plt.imshow(ldct, cmap="gray")
plt.subplot(1,2,2)
plt.title("Processed")
plt.imshow(proc, cmap="gray")
plt.show()
