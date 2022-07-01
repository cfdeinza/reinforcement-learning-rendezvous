# Useful `git` instructions:

## Remove cached files:
When you add a file to the `.gitignore` list, it will still show up in your repo. 
To stop tracking the file without actually deleting it from our local machine 
we need to remove it from the index. We can remove a single file using the following command:
```commandline
git rm --cached filename
```
To track the file again, use:
```commandline
git add filename
```

## Adding files from Colab:
After training, there should be three new (or updated) files:
- `models/best_model.zip`: The last model that was saved during training.
- `logs/evaluations.npz`: The average rewards obtained throughout the training.
- `logs/eval_trajectory_0.pickle`: The evaluated trajectory.

In the cell below, the three new files are added and committed to the local repository, 
and then the commit is pushed to the main repository on Github. If desired, the evaluated 
trajectory (`logs/eval_trajectory.pickle`) can also be saved in the same manner.

```python
# Add the files to git:

# The trained models:
!git add ./models/best_model.zip
!git add ./models/last_model.zip
# The rewards during training:
!git add ./logs/evaluations.npz
# The evaluated trajectory:
!git add ./logs/eval_trajectory_0.pickle
# Alternatively, to add all changes use: git add -all

# Commit:
!git commit -m 'Trained the model on Google colab'
# Push to github:
!git push origin main
```
