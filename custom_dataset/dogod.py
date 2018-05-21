gt_boxes = []
gt_classes = []



keys = self.targets[idx].keys()

for key in keys:
	if len(self.targets[idx][key]) == 0:
		continue

	elif len(self.targets[idx][key][0]) == 1:
		gt_boxes.append(self.targets[idx][key])
		gt_classes.append(self.class_map[key])

	else :
		for anns in self.targets[idx][key]:
			gt_boxes.append(anns)
			gt_classes.append(key)


def true_parse():
	{
	
	}