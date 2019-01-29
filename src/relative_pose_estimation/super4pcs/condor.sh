+Group = "GRAD"
+Project = "GRAPHICS_VISUALIZATION"
+ProjectDescription = "Transformation Synchronization"

Universe        = vanilla
requirements 	= InMastodon
Executable      = ./scannet/tasks.$(Process).sh
Output		= ./logs/$(Process).out
Error 		= ./logs/$(Process).err
Log		= ./logs/$(Process).log
arguments = $(Process) 120

Queue 120
