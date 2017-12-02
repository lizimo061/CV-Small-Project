%load('cameraParams.mat');
% list = dir('test*');
% imgarr = cell(length(list),1);
% for i = 1:length(list)
%     imgarr{i} = imread(strcat(list(i).name));
%     imgarr{i} = rgb2gray(imgarr{i});
%     imgarr{i} = undistortImage(imgarr{i},cameraParams);
% end
I = imgarr{1};
prevPoints = detectMinEigenFeatures(I, 'MinQuality', 0.001);

% Create the point tracker object to track the points across views.
tracker = vision.PointTracker('MaxBidirectionalError', 1, 'NumPyramidLevels', 6);

% Initialize the point tracker.
prevPoints = prevPoints.Location;
initialize(tracker, prevPoints, I);

% Store the dense points in the view set.
vSet = updateConnection(vSet, 1, 2, 'Matches', zeros(0, 2));
vSet = updateView(vSet, 1, 'Points', prevPoints);

% Track the points across all views.
for i = 2:numel(list)
    % Read and undistort the current image.
    I = undistortImage(imgarr{i}, cameraParams);

    % Track the points.
    [currPoints, validIdx] = step(tracker, I);

    % Clear the old matches between the points.
    if i < numel(imgarr)
        vSet = updateConnection(vSet, i, i+1, 'Matches', zeros(0, 2));
    end
    vSet = updateView(vSet, i, 'Points', currPoints);

    % Store the point matches in the view set.
    matches = repmat((1:size(prevPoints, 1))', [1, 2]);
    matches = matches(validIdx, :);
    vSet = updateConnection(vSet, i-1, i, 'Matches', matches);
end

% Find point tracks across all views.
tracks = findTracks(vSet);

% Find point tracks across all views.
camPoses = poses(vSet);

% Triangulate initial locations for the 3-D world points.
xyzPoints = triangulateMultiview(tracks, camPoses,...
    cameraParams);

% Refine the 3-D world points and camera poses.
[xyzPoints, camPoses, reprojectionErrors] = bundleAdjustment(...
    xyzPoints, tracks, camPoses, cameraParams, 'FixedViewId', 1, ...
    'PointsUndistorted', true);

colors = zeros(size(xyzPoints, 1), 3);
colored = cell(numel(imgarr), 1);
for i = 1:height(camPoses)
    img = imread(strcat(list(i).name));
    img = undistortImage(img,cameraParams);
    coloredPoints = colorizeBackProjection(xyzPoints, camPoses(i, :), cameraParams, img);
    colored{i} = coloredPoints;
    colors = colors + coloredPoints(:, 4:6);
end
    

% Average the colors
colors = colors ./ numel(imgarr) ./ 256;

figure(1);
plotCamera(camPoses, 'Size', 0.2);
hold on

% Exclude noisy 3-D world points.
goodIdx = (reprojectionErrors < 5);

% Display the dense 3-D world points.
pcshow(xyzPoints(goodIdx, :), colors(goodIdx, :), 'VerticalAxis', 'y', 'VerticalAxisDir', 'down', ...
    'MarkerSize', 45);
grid on
hold off

% Specify the viewing volume.
loc1 = camPoses.Location{1};
xlim([loc1(1)-5, loc1(1)+4]);
ylim([loc1(2)-5, loc1(2)+4]);
zlim([loc1(3)-1, loc1(3)+20]);
camorbit(0, -30);
axis equal;
title('Dense Reconstruction');