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
   *ACHTUNG VOR DEM SQUASHING DEN EDITOR UND ANDERE FILE SYSTEMS DIE DAZU GEHÖREN SCHLIEßEN!!!*
   > `git log`  
   > _STRG + EINFG um die nummer vom commit zu kopieren bis zu dem gestashed werden soll (nicht inkl.)_  
   > `git rebase -i <PRESS EINFG>`  
   > _Angezeigt werden die einzelnen commits mit mesagges_  
   > _ i drücken für insert _  
   > _alle bis auf den ersten `pick` mit `squash` ersetzen -> Squashed alle auf den ersten_  
   > _ESC : wq_  
   > _Neues Fenster ganz oben Gesamte commit message hinzufügen_  
   > `git log` 
   > _überprüfen ob alles geklappt hat_
   
