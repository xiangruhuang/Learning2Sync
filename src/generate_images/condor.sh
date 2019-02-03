+Group = "GRAD"
+Project = "GRAPHICS_VISUALIZATION"
+ProjectDescription = "Transformation Synchronization"

Universe        = vanilla
requirements 	= InMastodon
Executable      = ./run.sh
Output		= ./scannet_logs/$(Process).out
Error 		= ./scannet_logs/$(Process).err
Log		= ./scannet_logs/$(Process).log
arguments = $(Process) 100

Queue 100
