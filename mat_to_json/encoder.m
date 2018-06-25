mat = dir('*.mat');
for q = 1:length(mat)
    y = load(mat(q).name);
    labels = y.gTruth.LabelData;
    for j = 1:length(labels{1,:})
        for i = 1:length(labels{:,1})
            a = labels{:,:}{i,j};
            if ne(length(a),0)
                
                a(:,3) = a(:,1)+ a(:,3);
                y2 = a(:,2)+ a(:,4);
                a(:,4) = a(:,2);
                a(:,2) = y2;
                labels{:,:}{i,j} = a;
            end
        end
    end
    paths = y.gTruth.DataSource.Source;
    dict = [paths, labels];
end

text = jsonencode(dict);
timestamp = datestr(now,'YY-mm-DD-HH-MM');
filename = strcat('annotations-',datestr(now,'YY-mm-DD-HH-MM'),'.json');
fid = fopen(filename, 'w');
fwrite(fid, text, 'char');