+Group = "GRAD"
+Project = "GRAPHICS_VISUALIZATION"
+ProjectDescription = "Transformation Synchronization"

Universe        = vanilla
requirements 	= InMastodon
Executable      = ./compute_sigmas.sh
Output		= ./logs/$(Process).out
Error 		= ./logs/$(Process).err
Log		= ./logs/$(Process).log
arguments = $(Process)

Queue 100
