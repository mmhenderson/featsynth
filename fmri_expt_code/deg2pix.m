function p = deg2pix(p)
% converts degrees visual angle to pixel units before rendering
% with PTB. Needs p.screen_width_cm and p.vdist_cm
% js - 10.2007

% figure out pixels per degree, p.srect(1) is x coord for upper left of
% screen, and p.srect(3) is x coord for lower right corner of screen
if isfield(p,'screen_width_cm')
    p.ppd = pi * (p.srect(3)-p.srect(1)) / atan(p.screen_width_cm/p.vdist_cm/2) / 360;
elseif isfield(p,'screen_height_cm')
    p.ppd = pi * (p.srect(4)-p.srect(2)) / atan(p.screen_height_cm/p.vdist_cm/2) / 360;
else
    error('need either width or height of screen!')
end

% get name of each field in p
s = fieldnames(p);

% convert all fields with the word 'Deg' from degrees visual angle to
% pixels, then store in a renmaed field name
for i=1:length(s)
    ind = strfind(s{i}, '_deg');
    if ind
        curVal = getfield(p,s{i});
        tmp = char(s{i});
        newfn = [tmp(1:ind-1), '_pix'];
        p = setfield(p,newfn,curVal*p.ppd);
    end
end

end