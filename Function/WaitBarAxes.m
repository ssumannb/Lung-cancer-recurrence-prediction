classdef WaitBarAxes < handle
    %WaitBarAxes - display a wait bar within an axes.
    %   The WaitBarAxes class allows any axes object to be used as a Matlab
    %   style waitbar. This class also allows you to change the color of the
    %   waitbar.
    %
    %   To display multiple WaitBarAxes, use the MultiWaitBar class.
    %
    %   See also: MultiWaitBar.
    
    properties (Access = private)
        hAxes;          %Axes handle.
        hPatch;         %Patch handle.
        isVisible;      %True if wait bar is visible.
    end
    
    methods
        
        function self = WaitBarAxes(varargin)
            %WaitBarAxes - class constructor.
            %   obj = WaitBarAxes() returns an uninitialized WaitBarAxes object.
            %
            %   obj = WaitBarAxes(arg1, ...) also invokes the Initialize method with
            %   the provided argument(s).
            %
            %   See also: Initialize.
            if nargin > 0
                self.Initialize(varargin{:});
            end
        end
        
        function Initialize(self, hAxes, color)
            %Initialize - initialize the wait bar axes.
            %   obj.Initialize(hAxes) initializes axes to display a wait bar. The hAxes
            %   argument is the handle of an existing axes object.
            %
            %   obj.Initialize(self, hAxes, color) also sets the color of the wait bar,
            %   where the color argument is a valid ColorSpec (default = 'r').
            if ~exist('color', 'var')
                color = 'r';
            end
            self.hAxes = hAxes;
            set(self.hAxes, ...
                'Box','on',...
                'XLim',[0 1],...
                'YLim',[0 1],...
                'XTick',[],...
                'YTick',[],...
                'XTickLabel',[],...
                'YTickLabel',[], ...
                'Visible', 'off');
            self.hPatch = patch([0 0 0 0], [0 0 1 1], color, ...
                'EdgeColor', get(self.hAxes, 'XColor'), ...
                'EraseMode', 'normal', 'Visible', 'off');
            self.isVisible = false;
        end
        
        function Update(self, progress, axesTitle, color)
            %Update - update the wait bar.
            %   obj.Update(progress) updates the updates the wait bar axes. The value
            %   of progress must be a number in the range [0, 1].
            %
            %   obj.Update(progress, axesTitle) also updates the axes title.
            %
            %   obj.Update(progress, axesTitle, color) also updates the bar color,
            %   where the color argument is a valid ColorSpec (default = 'r').

            progress = max(0,min(progress, 1));
            xPatch = [0 progress progress 0];
            if exist('color', 'var')
                set(self.hPatch, 'XData', xPatch, 'FaceColor', color);
            else
                set(self.hPatch, 'XData', xPatch);
            end
            if nargin > 2 && ischar(axesTitle)
                title(self.hAxes, axesTitle);
            end
            if ~self.isVisible
                self.Show();
            end
            drawnow;
        end
        
        function Show(self)
            %Show - show the waitbar.
            %   obj.Show() makes the waitbar visible.
            set(self.hAxes, 'Visible', 'on');
            set(self.hPatch, 'Visible', 'on');
        end
        
        function Hide(self)
            %Hide - hide a waitbar.
            %   obj.Hide() hides the waitbar.
            set(self.hAxes, 'Visible', 'off');
            set(self.hPatch, 'Visible', 'off');
        end
        
    end
    
end

