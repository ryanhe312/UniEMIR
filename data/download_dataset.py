import tifffile,os,tqdm

# Download the dataset from the bossdb
from intern import array
dataset_path = 'bossdb://phelps_hildebrand_graham2021/FANC'
# dataset_path = 'bossdb://microns/minnie65_8x8x40'
group_path = 'em'
ddata = array(os.path.join(dataset_path,group_path))
print(ddata.shape)

# Download the dataset from the s3 bucket
# dataset_path = 's3://janelia-cosem-datasets/jrc_mus-kidney/jrc_mus-kidney.n5'
# dataset_path = 's3://janelia-cosem-datasets/jrc_mus-liver/jrc_mus-liver.n5'
# dataset_path = 's3://janelia-cosem-datasets/jrc_ctl-id8-2/jrc_ctl-id8-2.n5'
# group_path = 'em/fibsem-uint8/s0'

# import zarr
# group = zarr.open(zarr.N5FSStore(dataset_path, anon=True))
# zdata =  group[group_path] 
# import dask.array as da
# ddata = da.from_array(zdata, chunks=zdata.chunks)
# print(ddata.shape)

# set the output path
# output_path = '/home/user2/dataset/microscope/OpenOrganelle'
output_path = '/home/user2/dataset/microscope/FANC'
# output_path = '/home/user2/dataset/microscope/MICrONS'
output_path = os.path.join(output_path,dataset_path.split('/')[-1],group_path+'_tif')
os.makedirs(output_path,exist_ok=True)

if 'fibsem' in output_path:
    # em
    z_range = ddata.shape[0]//2-64,ddata.shape[0]//2+64 # liver, kidney
    x_range = ddata.shape[1]//2-1024,ddata.shape[1]//2+1024 # liver, kidney
    y_range = ddata.shape[2]//2+2048,ddata.shape[2]//2+4096 # liver
    # y_range = ddata.shape[2]//2-1024,ddata.shape[2]//2+1024 # kidney

elif 'FANC' in output_path:
    # fanc
    z_range = 1228-256,1228+256 
    x_range = 115965-256,115965+256
    y_range = 23124-256,23124+256

else:
    # microns-minnie
    z_range = 19000-256,19000+256
    x_range = 56298-256,56298+256
    y_range = 79190-256,79190+256

count = 0
for i in tqdm.tqdm(range(z_range[0],z_range[1])):
    # tifffile.imwrite(os.path.join(output_path,f"{count}.tif"), ddata[i,x_range[0]:x_range[1],y_range[0]:y_range[1]].compute()) # s3 bucket
    tifffile.imwrite(os.path.join(output_path,f"{count}.tif"), ddata[i,x_range[0]:x_range[1],y_range[0]:y_range[1]]) # bossdb
    count += 1