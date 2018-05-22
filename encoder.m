y = load('encode_me.mat');
text = jsonencode(y.gTruth.LabelData);
fid = fopen('encoded.json', 'w');
fwrite(fid, text, 'char');