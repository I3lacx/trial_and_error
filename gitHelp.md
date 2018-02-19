# Git Guide

This is a small Git guide to list and explain some basic concepts using git.
This file is not ment as a tutorial, its more of a guideline for me

`git checkout -- <file>`  
   Remove changes from your commit

`git checkout <commit-number>`  
Visit a branch to the time of the given commit

`git commit -m "<message>`  
Commit your current changes

`git status`  
Check what is and what isn't commited + check if there are new things from the repo

`git log`  
Check your old commits in a list

`git commit --amend`  
Modify your last commit message

`git reset HEAD~<number>`  
This will uncommit your old changes from the last <number> commits

## Squashing 
   __CLOSE EDITOR AND STOP THE SERVER!__
   > `git log`  
   > _ Press STRG + EINFG to copy the number of the commit you want to stash (not inclusive)_  
   > `git rebase -i <PRESS EINFG>`  
   > _Now you can see the single commits with their messages_  
   > _ press i for insert mode_  
   > _replace `pick` with `squash` or `s` to squash every commit onto the first one_  
   > _ESC : wq_  
   > _Now you can enter a new commit message for the whole commit_  
   > `git log` 
   > _to check if everything worked as planned_
   
