+Group = "GRAD"
+Project = "GRAPHICS_VISUALIZATION"
+ProjectDescription = "Transformation Synchronization"

Universe        = vanilla
requirements 	= InMastodon
Executable      = ./mat2obj.sh
Output		= ./logs/$(Process).out
Error 		= ./logs/$(Process).err
Log		= ./logs/$(Process).log
arguments = $(Process) 152

Queue 152
