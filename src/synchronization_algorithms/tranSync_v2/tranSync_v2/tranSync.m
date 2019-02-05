function tranSync(source, dataset)

proj_path = '/home/xrhuang/Projects/Learning2Sync';

n=100;
shapeids = dir(strcat([proj_path, '/processed_dataset/', dataset, '/*']));
mats = dir(strcat([proj_path, 'relative_pose/summary/', dataset, '/fgr/*.mat']));
mats
aaaaaa
avgrot = 0.0;
avgtrans = 0.0;
count = 0;
for s = shapeids'
    if strcmp(s.name, '.')
        continue;
    end
    if strcmp(s.name, '..')
        continue;
    end
    
    shapeid = s.name;
    mat = strcat([proj_path, '/relative_pose/summary/', dataset, '/', source, '/', shapeid, '.mat']);
    if ~(exist(mat, 'file') == 2)
        continue;
    end
    mat = load(mat);
    sig = mat.sigma;
    Tstar = mat.Tstar;
    T = mat.T;
    count = count + 1;
    Tcell = cell(n, n);
    Gcell = cell(n);
    for i = 1:n
        Gcell{i} = squeeze(Tstar(i, :, :));
    end
    for i = 1:n
        for j = i+1:n
%             Tcell{i, j} = (Gcell{j}*inv(Gcell{i}))';
            if sig(i, j) > 0.1
                Tcell{i, j} = zeros(4, 4);
                Tcell{j, i} = zeros(4, 4);
            else
                Tcell{i, j} = T((i*4-3):(i*4), (j*4-3):(j*4))';
                Tcell{j, i} = inv(Tcell{i, j});
            end
        end
        Tcell{i, i} = eye(4);
    end
    Tsync = synchroniseTransformationsZ(Tcell, 'rigid');
    
    [rot, trans] = avgerror(n, Tcell, Gcell, sig, 0.2);
    [rot2, trans2] = avgerror(n, Tsync, Gcell, sig, 0.2);
    [rot, trans, rot2, trans2]
    avgrot = avgrot + rot2;
    avgtrans = avgtrans + trans2;
end
avgrot = avgrot / count;
avgtrans = avgtrans / count;
[avgrot, avgtrans]
% n = 100;
% Tcell = cell(n, n);
% G = cell(n);
% for i = 1:n
%     [R,~]=qr(randn(3));
%     t = randn(3, 1);
%     Ti = zeros(4, 4);
%     Ti(1:3, 1:3) = R;
%     Ti(1:3, 4) = t;
%     Ti(4, 4) = 1.0;
%     G{i} = Ti;
% end
% 
% for i = 1:n
%     for j = 1:n
%         Tcell{i, j} = (G{j}*inv(G{i}))';
%         Tcell{i, j}(1:4, 1:3) = Tcell{i, j}(1:4, 1:3) + randn(4, 3) * 0.01;
%     end
% end
% Tsync = synchroniseTransformationsZ(Tcell, 'rigid');
%         
% avgerror(n, Tcell, G)
% avgerror(n, Tsync, G)

end

function [v] = readRelPose(mat_path)
   loadmat(mat_path);
   v = vertex;
end

function [T] = readGT(n, mat_path)
    T = cell(n);
    for i = 0:(n-1)
        mat_file = strcat([mat_path, '/', int2str(i), '.mat'])
        T{i} = load(mat_file, 'pose');
    end
end

function [rot, trans] = avgerror(n, T, G, sigma, threshold)

rot = 0;
trans = 0;
count = 0;
for i = 1:n
    for j = i+1:n
        if sigma(i, j) > threshold
            continue;
        end
        count = count + 1;
        Tij = T{i, j}';
        Gij = (G{j}*inv(G{i}));
        trans = trans + norm(Tij(1:3, 4)-Gij(1:3, 4), 2);
        tra = (trace(Tij(1:3, 1:3)'*Gij(1:3, 1:3)) - 1.0)/ 2.0;
        if tra > 1.0
            tra = 1.0
        else
           if tra < -1.0
               tra = -1.0
           end
        end
        rot = rot + acos(tra);
        % e = e + norm(T{i, j} - (G{j}*inv(G{i}))', 'fro')^2;
    end
end

rot = (rot / count) / pi * 180.0;
trans = trans / count;

end
