# Unveiling Digital Mirrors

This repository provides the data of the analysis for the paper [LINK TO PAPER]() published in [LINK TO JOURNAL]().

Additionally, it also contains the 150 prototypes computed from the clusters in the paper for extra transparancy of the analysis conducted in the paper. These are packaged as 150 .png-Files in the `Prototypes.zip' archive.

The data provided is already filtered and labelled as described in the 'Data and Methods' section 3.1 of the paper.

The data set is provided as a .csv (separator `;`) which contains information on the coordinates of the keypoints extracted by [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).

As described in the paper, the data contains 15167 cases which are distributed across gender as follows:

| Gender | Frequency | Relative frequency | 
| --- | --- | --- |
| Female | 7957 | 0.5246 |
| Male | 6597 | 0.4350 |
| Non-Binary | 613 | 0.0404 |

The data contains 35 colums of which 34 contain the information on keypoints. The remaining column contains the gender the image has been annotated with.

Keypoints on the horizontal axis are prefixed with `x_` and keypoints on the vertical axis are prefixed with `y_`. The names of the keypoints are in accordance with the [specification of the OpenPose API](https://cmu-perceptual-computing-lab.github.io/openpose/web/html/doc/md_doc_02_output.html).

For example: `x_nose` and `y_nose` contain the x- and y-coordinate of the pixel at which the nose has been detected. Note that at this point, the coordinates have not yet been normalized, but represent actual values of pixels.