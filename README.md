# SmartStitcher
---
A stitching software that can quickly stitch **terabytes** of 3D pictures, and supports the output of results in **terafly** format.(2023.5.11)

---

## requirement
+ python==3.9.12
+ QtPy==2.0.1
+ tifffile==2022.5.4
+ opencv-python==4.6.0.66
+ math
+ time
+ numpy==1.23.1
+ skimage
+ sklearn
+ PIL
+ matplotlib
+ multiprocessing
---
## User's guide
![Main page](readme_res\Main_page.png
"Main page")

### Input
>Currently the program only supports **TIFF** format **tiles** file data

SmartStitcher input can be selected in two modes: 
1. Direct import of JSON files containing data information. 
2. Enter the necessary information of data data in the box (the content framed in the red box), and click the `saveJson` button, the json file will be automatically generated and saved in the **output path** filled in.

The meaning of each input box is explained below:
+ `input`: the address where the low-resolution data is stored
+ `high_res`: The address where the high-resolution data is stored
>! If the original data does not have multiple resolutions from, the contents of the two input boxes can be consistent, but cannot be empty. And the file names of the corresponding files of the two resolutions should be exactly the same.
+ `output`: The paths where the various intermediate files and the final result are stored
+ `highest_vixel_size`: Size ratio of highest resolution to highest resolution **(must be 1)**
+ `x_length` `y_length` `z_length`: The pixel size of the **highest** resolution
+ `overlapX` `overlapY` `overlapZ`: Theoretical overlap between adjacent pictures to be stitched together **(value less than 1)**
+ `wildname`: This item has no effect and remains empty
+ `row` `col`:The number of rows and columns with a stitched picture population
>! At any stage of the stitching, we can click the `print json` button and display the information of the current json file in the space below to check the correctness of the stitching.
>>! The program will not automatically overwrite or overwrite the information of the json file on disk, and when you need to save the current stitching information, you need to manually click the `saveJson` button.
---
### Set Arrangement
+After you have imported the initial information of the original data, click the first button to set the file arrangement.
![arrangement](readme_res\arrangement.png)
In the pop-up subwindow, select how the files are arranged in the folder.

![arangement dialog](readme_res\arangementdiaglog.png)

Below I will explain each arrangement separately:
+ `row-by-row`: Folders are arranged in folders in the order of the rows in the stitching.
>D&R:From top to bottom, left to right

>D&L:From top to bottom, right to left

>The rest will not be repeated...

+ `column-by-column`: Similar to `row-by-row`
+ `snake-by-row`: Arranged in a serpentine manner in row order.
+ `snake-by-col`: Arranged in a serpentine manner in colunm order.

After submission, an information item with a key value of locations will be generated in the JSON file to save the arrangement information.

---
### Generate low-resolution MIPs
Click the `createMIP_low_res` button

---
### Preview MIPresult and classify:
1. Click the `MIP preview` button on the main page
2. Click `create_preview` generate a preview of the effect before stitching in the pop-up sub-window to check whether the overlap arrangement is set correctly.
>! The generated preview file will be saved automatically, and the generated image will not have to be recalculated when browsing again.

3. On the picture display page on the left, select positive samples (green) by left-clicking with the mouse, select negative samples (red) with the right mouse button, select more than 15 samples each, and click `classify` on the right to classify.
>![classify](readme_res\claasify.png)
4. Wait for the classification run to finish, the result of the classification will be displayed on the left side. For isolated points in the result, they can be connected to the subject via the `Connect` button, or they can be eroded into a negative sample by clicking `Erode`. In addition, it is possible to use the mouse click to make a more precise secondary selection.
> If you are not satisfied with the results, you can click the refresh button to refresh the classification results and reclassify them.

5. Click `commit` to submit the classification results. (The classification results will determine the stitching strategy for the highest resolution data)
>! If you don't want to filter, you can stitch all the tiles that are kept, and you can click `commit` without clicking `classify`

---
### Calculation of low-resolution offsets
>! If you only have a single resolution, you can skip this step.

Click the `4.low_res_x_y_shift` button and `5.low_res_z_shift` button and wait for the progress bar to reach 100%. **(Both buttons can be clicked at the same time, and the task is synchronized with multiple threads.)**

---

 ### Generate high-resolution MIPs
 Click the `6.createMIP_high_res` button and wait for the progress bar to finish.
 >! This step will only generate MIPs that were previously filtered as positive

 ---

### Calculation of high-resolution offsets
Click the `7.high_res_x_y_shift` button and `8.high_res_z_shift` button and wait for the progress bar to reach 100%. **(Both buttons can be clicked at the same time, and the task is synchronized with multiple threads.)**

---
### Global optimization 
1. High-resolution data: Click the `9.high_res_optimazation` button and wait for the progress bar to end.
2. Low-resolution data: **！ It must be done after the `9.high_res_optimazation`.**  Click the 10.low_res_optimazation button and wait for the progress bar to finish.(This step can be skipped for a single resolution)
---
### Float To Int
Click `11.HR-shift_floatToint` and `12. LR-shift_floatToint` button, which converts the global optimization results of high and low resolution into global coordinate values of integers, respectively.
>! Global optimization must be performed to perform this step (if a single resolution skips global optimization for low resolution, this step cannot be performed at low resolution)
---
### Output
>The following operations are in no particular order

+ *Generate MIP in the z-direction of the stitching result*: Click the `LR-stitched-MIP` or `HR-stitched-MIP` button, the results will be named after the MIP_stitched.tiff and stored in the output path\HR(LR)_result_slice.

+ *Output the results in z-slice form*: Click the `LR-z_slice-result` or `HR-z_slice-result` button

+ *Output multi-level resolution results in Telafly format*:
    1. Select the resolution you want to output in the GroupBox in the upper right corner of the main page.
    >! If the highest resolution is classified, only the `part_stitch` can be selected when the resulting output is larger than the lowest resolution.

    >The resolution does not have to be selected all
    >![teraflyoutput](readme_res\teraflyoutput.png)

    2.Click the `output_terafly` button and wait for the progress bar to finish.  
---
At this point, the whole splicing process is completed.

 of course, if you slice relative to the original data, you can click the `make_slice_LR` or `make_slice_HR button`. (Takes too long)




