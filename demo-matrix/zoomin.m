function pan = zoomin(ax,areaToMagnify,panPosition)
% AX is a handle to the axes to magnify
% AREATOMAGNIFY is the area to magnify, given by a 4-element vector that defines the
%      lower-left and upper-right corners of a rectangle [x1 y1 x2 y2]
% PANPOSTION is the position of the magnifying pan in the figure, defined by
%        the normalized units of the figure [x y w h]
%

fig = ax.Parent;
pan = copyobj(ax,fig);
pan.Position = panPosition;
pan.XLim = areaToMagnify([1 3]);
pan.YLim = areaToMagnify([2 4]);
pan.XTick = [];
pan.YTick = [];
rectangle(ax,'Position',...
    [areaToMagnify(1:2) areaToMagnify(3:4)-areaToMagnify(1:2)]); hold on;
xy = ax2annot(ax,pan.XLim,pan.YLim,pan.YScale);
annotation(fig,'line',[xy.XLim(1) panPosition(1)],...
    [xy.YLim(2) panPosition(2)],'Color','k');hold on;
annotation(fig,'line',[xy.XLim(2) panPosition(1)+panPosition(3)],...
    [xy.YLim(2) panPosition(2)],'Color','k');hold on;

pan.Title.Visible = 'off';
pan.XLabel.Visible = 'off';
pan.YLabel.Visible = 'off';
pan.XTick = pan.XLim;
pan.FontSize = 8;
end

function anxy = ax2annot(ax,XLim,YLim,type)
% This function converts the axis unites to the figure normalized unites
% AX is a handle to the figure
% XY is a n-by-2 matrix, where the first column is the x values and the
% second is the y values
% ANXY is a matrix in the same size of XY, but with all the values
% converted to normalized units

pos = ax.Position;
aXLim = ax.XLim;
aYLim = ax.YLim;
if strcmp(type, 'log')
    YLim = log10(YLim);
    aYLim = log10(aYLim);
end
%   white area * ((value - axis min) / axis length)   + gray area
anxy.XLim = pos(3)*((XLim-aXLim(1))./range(aXLim))+ pos(1);
anxy.YLim = pos(4)*((YLim-aYLim(1))./range(aYLim))+ pos(2);
end