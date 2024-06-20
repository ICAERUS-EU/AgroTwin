import pandas as pd
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os
import shapefile as shp
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter
from osgeo import gdal, osr, ogr
import fiona
import rasterio
from rasterio.plot import show
import earthpy.plot as ep
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

# INPUT PARAMETERS
shape_file = "Borders.shp" #Field borders shp format
excel_input = pd.read_excel("Input_coordinates.xlsx", sheet_name='Sheet1') #select spreadsheet with coordinates UTM32N path
cloud_input = "Input_pointcloud.ply" #select pointcloud path
input_csv = "Results.csv" #Results in lat lon coordinates
delta = 2.8  # inter-row (distance between two rows )
x_max_vine = 0.8  # vine end point (cordon length)
z_min_vine = 0.7  # trunk height from ground

# Uncomment for selecting the phenological phase and the sprayer type
#BBCH = "10-55" 
BBCH = "55-71"  
#BBCH == "71-89" 

#SPRAYER_TYPE = "standard"  # conventional sprayer
#SPRAYER_TYPE = "deflectors"
SPRAYER_TYPE = "multi spout"
#SPRAYER_TYPE = "vertical booms"
#SPRAYER_TYPE = "recycling tunnel"


# FIXED VALUES
x_min_vine = 0  # vine start point 
y_min_vine = -1  # fixed value
y_max_vine = 1  # fixed value
z_max_vine = 10  # fixed value


output_dir = "Maps"
os.makedirs(output_dir, exist_ok=True) # Create the directory if it doesn't exist

#FUNCTIONS

def readCSV(csvFile):
    return pd.read_csv(csvFile, delimiter = ',', header = 0) 

def readSHP(shapeFile):
    return shp.Reader(shapeFile)

def latMin(shapeData):
    return shapeData.shape(0).bbox[1]

def latMax(shapeData):
    return shapeData.shape(0).bbox[3]

def lonMin(shapeData):
    return shapeData.shape(0).bbox[0]

def lonMax(shapeData):
    return shapeData.shape(0).bbox[2]

def toDict(shapeData):
    myDictionary = {}
    myDictionary["latMin"] = latMin(shapeData)
    myDictionary["lonMin"] = lonMin(shapeData)
    myDictionary["latMax"] = latMax(shapeData)
    myDictionary["lonMax"] = lonMax(shapeData)
    return myDictionary
    
def distance_matrix(x0, y0, x1, y1):

    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    d0 = np.subtract.outer(obs[:,0], interp[:,0])
    d1 = np.subtract.outer(obs[:,1], interp[:,1])
    
    # calculate hypotenuse
    return np.hypot(d0, d1)
    
def interpolation(x, y, z, xi, yi, power=4, smoothing_sigma=70):
    
    dist = distance_matrix(x, y, xi, yi)
    
    weights = 1.0 / (dist)**power

    weights /= weights.sum(axis=0)

    interpolated_values = np.dot(weights.T, z)

    interpolated_values = gaussian_filter(interpolated_values, smoothing_sigma)

    return interpolated_values   
    

def saveReclassedSHP(inputTiff, outputSHP, valueClass):
    raster = gdal.Open(inputTiff, gdal.GA_Update)

    band = raster.GetRasterBand(1)
    proj = raster.GetProjection()

    shp_proj =osr.SpatialReference()
    shp_proj.ImportFromWkt(proj)
    
    call_drive = ogr.GetDriverByName('ESRI Shapefile')
    
    create_shp = call_drive.CreateDataSource(outputSHP)
    
    shp_layer = create_shp.CreateLayer('layername', srs = shp_proj)
    new_field = ogr.FieldDefn(str('ID'), ogr.OFTInteger)
    
    val_field = ogr.FieldDefn("Value", ogr.OFTReal)
    shp_layer.CreateField(val_field)
    
    if valueClass == "LAI":
        poly_list = gdal.FPolygonize(band, None, shp_layer, 0, [], callback = None)
    elif valueClass == "V_r":
        poly_list = gdal.Polygonize(band, None, shp_layer, 0, [], callback = None)
    else:
        raise ValueError("valueClass must be either LAI or V_r")
    
    create_shp.Destroy()
    raster = None
    
    # Open the shapefile in reading mode
    ds = ogr.Open(outputSHP, 1)
    layer = ds.GetLayer()
    
    # Select specific value
    value_to_remove = 0
    
    for i in range(layer.GetFeatureCount()):
        feature = layer.GetFeature(i)
        poly = feature.GetGeometryRef()
        value = feature.GetField("Value")
        
        if value == value_to_remove:
            layer.DeleteFeature(i)

    ds = None

def idw_with_map(lat_list, lon_list, value_list, shapePath, title):

    shapeData = readSHP(shapePath)

    lat_min = latMin(shapeData)
    lat_max = latMax(shapeData)
    lon_min = lonMin(shapeData)
    lon_max = lonMax(shapeData)
    
    lon_arr = np.linspace(lon_min, lon_max, 1000)
    lat_arr = np.linspace(lat_min, lat_max, 1000)
    lon_mesh, lat_mesh = np.meshgrid(lon_arr, lat_arr)
    
    # Normalize data for iterpolation
    value_list_normalized = (value_list - np.min(value_list)) / (np.max(value_list) - np.min(value_list))
    
    # IDW + gaussian filter
    data_mesh_normalized = interpolation(np.asarray(lon_list), np.asarray(lat_list), np.asarray(value_list_normalized), lon_mesh.flatten(), lat_mesh.flatten())

    # Reshape matrix for visualization
    data_mesh_normalized = data_mesh_normalized.reshape((len(lat_arr), len(lon_arr)))

    ### Graphic
    url = 'http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}'
    stamen_terrain = cimgt.GoogleTiles(url=url) 

    # Create a GeoAxes in the tile's projection.
    fig = plt.figure(figsize=(12,10))
    fig.set_facecolor("white")
    ax = fig.add_subplot(1, 1, 1, projection=stamen_terrain.crs)

    ax.set_aspect('equal')
  
    # Limit the extent of the map to a small longitude/latitude range.
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Add the Stamen data at zoom level 8.
    ax.add_image(stamen_terrain, 20)

    x, y = lon_mesh, lat_mesh

    # De-normalization for visualization
    data_mesh = data_mesh_normalized * (np.max(value_list) - np.min(value_list)) + np.min(value_list)

    z_min = min(value_list)
    z_max = max(value_list)

    prev_ext = ax.get_extent()
    ax.set_extent([x.min(), x.max(), y.min(), y.max()], crs=ccrs.PlateCarree())
    ax.imshow(np.flip(data_mesh, axis=0), origin='upper', interpolation='gaussian', alpha=0.7, cmap='Blues', extent=ax.get_extent(), zorder=5, vmin=z_min, vmax=z_max)
    mesh = ax.imshow(np.flip(data_mesh, axis=0), origin='upper', interpolation='gaussian', alpha=1, cmap='Blues', extent=ax.get_extent(), zorder=-2, vmin=z_min, vmax=z_max)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=1, color='black', alpha=0.7, linestyle='--', draw_labels=True)
    gl.top_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = False
    gl.xlabel_style = {'rotation': 25, 'ha': 'right'}
    gl.ylabel_style = {'rotation': 25, 'ha': 'right'}

    ax.set_title(title, fontsize=30)
    fig.colorbar(mesh)
    fig.tight_layout()
    
    fileNAME = f'{title}.png'
    fileNAME = os.path.join(output_dir, os.path.basename(fileNAME))
    fig.savefig(fileNAME, bbox_inches='tight', pad_inches=0.3)
       
    return data_mesh


def getGeoTransform(extent, nlines, ncols):
    resx = (extent[2] - extent[0]) / ncols
    resy = (extent[3] - extent[1]) / nlines
    return [extent[0], resx, 0, extent[3] , 0, -resy]

def meshToGeoTIFF(idwMesh, shapeData, filename):

    # Flipping map for correct visualization
    idwMeshFlip = np.fliplr(np.flip(idwMesh))
     
    # Data extension (min. lon, min. lat, max. lon, max. lat)
    myBBox = toDict(shapeData)
    extent = [myBBox["lonMin"], myBBox["latMin"], myBBox["lonMax"], myBBox["latMax"]] 
     
    # Get GDAL driver GeoTiff
    driver = gdal.GetDriverByName('GTiff')
     
    # Get dimensions
    nlines = idwMeshFlip.shape[0]
    ncols = idwMeshFlip.shape[1]
    data_type = gdal.GDT_Float32
    
    grid_data = driver.Create('grid_data', ncols, nlines, 1, data_type)
     
    # Write data for each bands
    grid_data.GetRasterBand(1).WriteArray(idwMeshFlip)
     
    # Lat/Lon WSG84 Spatial Reference System
    srs = osr.SpatialReference()
    srs.ImportFromProj4('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
     
    # Setup projection and geo-transform
    grid_data.SetProjection(srs.ExportToWkt())
    grid_data.SetGeoTransform(getGeoTransform(extent, nlines, ncols))
     
    # Save the file
    print(f'Generated GeoTIFF: {filename}')
    driver.CreateCopy(filename, grid_data, 0)  
     
    # Close the file
    driver = None
    grid_data = None
     
    # Delete the temp grid
    os.remove('grid_data')
    return

def rasterCrop(tmpTiffPath, shapePath, fileName):

    with fiona.open(shapePath, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]
        
    print('shapes: ', shapes)

    with rasterio.open(tmpTiffPath) as src:
        print('contenuto: ',src.read(1))
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta
    

    out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform})

    with rasterio.open(fileName, "w", **out_meta) as dest:
        dest.write(out_image)
    return

# Reclassify Geotiff 
def reclassify(tmpTiffName, outTiffName):
    ds = rasterio.open(tmpTiffName)
    show(ds)
    
    data = ds.read()
    print('data max', data.max())
    print('data min', data.min())
    
    lista = np.asarray(data.copy())
    mask = lista != 0
    qlista = lista[mask]
    
    # Calculate quantiles
    quantiles = np.quantile(qlista, q=[0.25, 0.75, 1])
    quantiles[2] += 1 # +1 to include max val
    print('quantili', quantiles)
    
    # Divide the data into classes based on quantiles
    class_data = np.digitize(lista, quantiles, right=False)
    
    class_data = np.float32(class_data)
    
    print("min", class_data.min())
    print("max", class_data.max())
    print(class_data)
    
    # Assign the value of the quantile to each class
    classMean =    [(quantiles[0] + qlista.min()) / 2,
                    (quantiles[1] + quantiles[0]) / 2,
                    (qlista.max() + quantiles[1]) / 2]

    class_data[np.where(class_data==0)] = round(classMean[0], 2)
    class_data[np.where(class_data==1)] = round(classMean[1], 2)
    class_data[np.where(class_data==2)] = round(classMean[2], 2)
    
    with rasterio.open(outTiffName, 'w',
                        driver=ds.driver,
                        height=ds.height,
                        width=ds.width,
                        count=ds.count,
                        crs=ds.crs,
                        transform=ds.transform,
                        dtype=class_data.dtype
                        ) as dst:
        dst.write(class_data)

    return classMean

def saveReclassedPNG(tmpTiffName, classMean, pngFilename, valueClass):
    
    # Visualization of cutted Geotiff 
    pngFilename = os.path.join(output_dir, os.path.basename(pngFilename))

    # Open file geotiff
    with rasterio.open(tmpTiffName) as src:
        geotiff = src.read()

        # Converting in float32 and replace the zeros with nans
        geotiff = np.float32(geotiff)
        geotiff[np.where(geotiff == 0)] = np.nan
    
    # Create a figure
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Create the map with Google Tiles
    url = 'http://mt0.google.com/vt/lyrs=s&hl=en&x={x}&y={y}&z={z}'
    stamen_terrain = cimgt.GoogleTiles(url=url)
    ax.add_image(stamen_terrain, 19)
    
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.5, color='black', alpha=0.5, linestyle='--', draw_labels=True)
    gl.top_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = False
    gl.xlabel_style = {'size': 15,'rotation': 25, 'ha':'right'}
    gl.ylabel_style = {'size': 15,'rotation': 25, 'ha':'right'}
    
    if valueClass == "V_r":
        ax.set_title("V [l/ha]", fontsize = 25)
        # Overlay the geotiff data on the map
        im = ax.imshow(geotiff[0], extent=rasterio.plot.plotting_extent(src), origin='upper', alpha=0.5, zorder=10, transform=ccrs.PlateCarree(), cmap='Blues')
        ep.draw_legend(im, titles=[str("{:.0f}".format(classMean[0])), str("{:.0f}".format(classMean[1])), str("{:.0f}".format(classMean[2]))])
    elif valueClass == "LAI":
        ax.set_title("LAI [m^2/m^2]", fontsize = 25)
        # Overlay the geotiff data on the map
        im = ax.imshow(geotiff[0], extent=rasterio.plot.plotting_extent(src), origin='upper', alpha=0.5, zorder=10, transform=ccrs.PlateCarree(), cmap='Greens')
        ep.draw_legend(im, titles=[str("{0:.2f}".format(classMean[0])), str("{0:.2f}".format(classMean[1])), str("{0:.2f}".format(classMean[2]))])
    else:
        raise ValueError("valueClass must be either LAI or V_r")
    
    fig.savefig(pngFilename, bbox_inches = 'tight', pad_inches = 0.3)
    return

def drawMap(dataRaster, valueClass, shape, basename):
    if valueClass == "LAI":
        basename = basename + "_LAI"
    elif valueClass == "V_r":
        basename = basename + "_V_r"
    else:
        raise ValueError("valueClass must be either LAI or V_r")

    tmpTiffName = os.path.join(output_dir, basename + "_tmp.tif")
    tmpTiffName2 = os.path.join(output_dir, basename + "_tmp_2.tif")
    cropTiffName = os.path.join(output_dir, basename + "_crop.tif")
    outTiffName = os.path.join(output_dir, basename + ".tif")

    shapeData = readSHP(shape)

    meshToGeoTIFF(dataRaster, shapeData, tmpTiffName)

    rasterCrop(tmpTiffName, shape, cropTiffName)
    
    reclassTiffName = os.path.join(output_dir, basename + "_reclass.tif")
    intervals = reclassify(cropTiffName, reclassTiffName)

    rasterCrop(reclassTiffName, shape, tmpTiffName2)

    pngName = os.path.join(output_dir, basename + ".png")
    shpName = os.path.join(output_dir, basename + ".shp")
    saveReclassedPNG(tmpTiffName2, intervals, pngName, valueClass)
    saveReclassedSHP(tmpTiffName2, shpName, valueClass)
    

# ANALYSIS
Results_list = []  # List to store results for each vine

for i in range(len(excel_input)):
    ID = excel_input.iloc[i, 0]  # Identify vine ID
    a_lat = excel_input.iloc[i, 1]  # Identify A latitude coordinate
    a_lon = excel_input.iloc[i, 2]  # Identify A longitude coordinate
    b_lat = excel_input.iloc[i, 3]  # Identify B latitude coordinate
    b_lon = excel_input.iloc[i, 4]  # Identify B longitude coordinate

    # Read original full vineyard point cloud
    point_cloud = o3d.io.read_point_cloud(cloud_input)
    o3d.visualization.draw_geometries([point_cloud], window_name="Vineyard Point Cloud")
    colors = np.asarray(point_cloud.colors) # Save colors in array 

    # Define line passing through start (a) and end (b) points of the vine portion
    if b_lon < a_lon:
        m = (b_lon - a_lon) / (b_lat - a_lat)  # Angular coefficient of the line passing from A and B
    else:
        m = -(b_lon - a_lon) / (b_lat - a_lat)  # Angular coefficient of the line passing from A and B
    
    angle = np.arctan(m)

    # Translate point cloud to vine portion coordinates
    translation_matrix = np.array([[1, 0, 0, -a_lat], [0, 1, 0, -a_lon], [0, 0, 1, 0], [0, 0, 0, 1]])
    point_cloud.transform(translation_matrix)

    # Rotate point cloud to align vine rows to x-axis
    rotation_matrix = Rotation.from_euler('z', angle).as_matrix().transpose()
    ptCloudRotateR = (np.matmul(np.asarray(point_cloud.points), rotation_matrix))  
    
    # Vine portion extraction
    vine_portion = [x_min_vine, x_max_vine, -(2 * 2.8 / 3), (2 * 2.8 / 3), -1000, 1000]  # Vine Portion bounds
    indices = np.where(
        (ptCloudRotateR[:, 0] >= vine_portion[0]) & (ptCloudRotateR[:, 0] <= vine_portion[1]) &
        (ptCloudRotateR[:, 1] >= vine_portion[2]) & (ptCloudRotateR[:, 1] <= vine_portion[3]) &
        (ptCloudRotateR[:, 2] >= vine_portion[4]) & (ptCloudRotateR[:, 2] <= vine_portion[5])
    )[0]
    ptCloudVinePortion = ptCloudRotateR[indices]

    # Visualize the point cloud of the selected vine
    vine_portion_point_cloud = o3d.geometry.PointCloud()
    vine_portion_point_cloud.colors = o3d.utility.Vector3dVector(colors[indices])
    vine_portion_point_cloud.points = o3d.utility.Vector3dVector(ptCloudVinePortion)
    o3d.visualization.draw_geometries([vine_portion_point_cloud], window_name="Vine Portion Point Cloud extracted")

""" 
1. SOIL MODELING

Calculate the soil shape in the proximity of the extracted vine. 
In the case of terracing, two adjacent inter-row paths can have different elevations. 
We need to find the relative heights of the vine points with respect to a plane that approximates the soil surface.

Analysis:
Select points from the point cloud that fall within specific ranges in the y and z dimensions (positive and negative). 
The selection criteria are combined into a Boolean mask, and then the points that meet these criteria are extracted.
Assess which subset of soil points is lower between the one to the right and the one to the left of the vine. 
Define the objective function to be minimized to find the best-fit plane that approximate the soil shape.
Calculate the relative heights of vine portion point cloud with respect to soil plane.


2. CANOPY DENSITY

Calculate the canopy density by dividing the canopy region into a grid and computing a density descriptor based on a threshold.

Analysis:
Setting the grid dimensions
Define the region of interest (ROI) for the vine canopy in the x, y, and z dimensions
Calculate density in each grid cell

3. CANOPY THICKNESS, HEIGTH AND VOLUME 

Calculate the canopy thickness, height and volume.

Analysis:
Extracting thickness height and thickness based on statistical analysis on canopy 3D point cloud
Evaluating canopy volume using volumetric algorithms

4. OPTIMUM VOLUME RATE

Calculate the optimal water volume rate for pesticide treatments.

Analysis:
Find the appropriate height/thickness dosing value and efficiency factors based on the canopy height/width
Determine Phenological Stage factor and Sprayer Performance Factors
Calculate outputs (LAI, TRV, LWA, Volume rate) """

# RESULTS
result_data = {
    'ID': ID,
    'a_lat': a_lat,
    'a_lon': a_lon,
    'b_lat': b_lat,
    'b_lon': b_lon,
    't': t,
    'h': h,
    'h_canopy': h_canopy,
    'canopy_vol': canopy_vol,
    'LAI': LAI,
    'LWA': LWA,
    'TRV': TRV,
    'V': V
}
Results_list.append(result_data)
print('Plant:', ID, 'Processed correctly', 'LAI', LAI, 'Vol', V)


# Write the DataFrame to a CSV file
Results = pd.DataFrame(Results_list) # Convert the list of dictionaries to a DataFrame
Results.to_csv('Results.csv', index=False) 

# GENERATE MAPS
csv_data = readCSV(input_csv)

# Esegui l'interpolazione e salva i risultati
raster_lai = idw_with_map(a_lat, a_lon, csv_data["LAI"].values, shape_file, "LAI")
raster_vr = idw_with_map(a_lat, a_lon, csv_data["V"], shape_file, "V_r")

drawMap(raster_vr, "V_r", shape_file, "input")
drawMap(raster_lai, "LAI", shape_file, "input")
   

