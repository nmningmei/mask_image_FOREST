#TO RUN: 
#place the code "make_labels.sh" into the folder that contains the subjectfolders
#type: sh make_labels.sh subjdir(*this is the subj folder that contains #mri and label subfolders*) 
sub=$1
names=(fusif infpar superiorparietal inftemp latoccip lingual phipp pericalc precun sfrontal parsoper parsorbi parstri middlefrontal) 

index=0

for i in 1007 1008 1029 1009 1011 1013 1016 1021 1025 1028 1018 1019 1020 1027;
do
    mri_binarize --i $sub/mri/aparc+aseg.mgz --match ${i} --o $sub/label/lh_`echo ${names[$index]}`.nii.gz 
    index=`expr $index + 1`
done

for mask in $sub/label/*.nii.gz;
do 
    if [[ "$mask" != *_fsl* ]]; then
        echo $mask
        mask_name=$(basename "$mask")
        new_path="$(dirname "$mask")/"${mask_name%.*.*}"_fsl.nii.gz"
        fslswapdim $mask x z -y "$new_path" # depends on structural being flipped or not
        mri_convert -i "$new_path" -rl $sub/mri/output.mnc -o "$new_path"
    fi
done

names=(fusif infpar superiorparietal inftemp latoccip lingual phipp pericalc precun sfrontal parsoper parsorbi parstri middlefrontal) 

index=0

for i in 2007 2008 2029 2009 2011 2013 2016 2021 2025 2028 2018 2019 2020 2027;
do
    mri_binarize --i $sub/mri/aparc+aseg.mgz --match ${i} --o $sub/label/rh_`echo ${names[$index]}`.nii.gz 
    index=`expr $index + 1`
done

for mask in $sub/label/*.nii.gz;
do 
    if [[ "$mask" != *_fsl* ]]; then
        echo $mask
        mask_name=$(basename "$mask")
        new_path="$(dirname "$mask")/"${mask_name%.*.*}"_fsl.nii.gz"
        fslswapdim $mask x z -y "$new_path" # depends on structural being flipped or not
        mri_convert -i "$new_path" -rl $sub/mri/output.mnc -o "$new_path"
    fi
done
 
