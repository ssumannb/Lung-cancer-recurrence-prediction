classdef MultiWaitBar < handle
    %MultiWaitBar - display multiple WaitBarAxes in a single window.
    %   The MultiWaitBar class allows the display of multiple wait bars in a
    %   one or two-dimensional grid within a single figure window. Each wait
    %   bar is an instance of the WaitBarAxes class. The appearance of the bar
    %   may be modified by changing its color from the default value of red.
    %
    %   Each bar within the grid is an instance of the WaitBarAxes class and
    %   can be passed to other functions or classes for updates. This allows a
    %   function to update a bar without knowledge of its position in the
    %   figure.
    %
    %   See also: WiatBarAxes.
    
    properties (Access = private)
        hFig;           %Figure handle.
        waitBarAxes;    %array of WaitBarAxes objects.
        isVisible;      %True if wait bar is visible.
        rowCount;       %Rows of wait bars.
        colCount;       %Columns of wait bars.
        w;              %width of single wait bar (points).
        h;              %height of single wait bar (points).
    end
        
    methods
        
        function self = MultiWaitBar(varargin)
            %MultiWaitBar - class constructor.
            %   obj = MultiWaitBar() creates an uninitialized MultiWaitBar object.
            %
            %   obj = MultiWaitBar(arg1, ...) also invokes the Initialize method with
            %   the provided arguments.
            %
            %   Example:
            %
            %       mwb = MultiWaitBar(3, 2, 'My wait bars', 'c');
            %       mwb.Update(1, 1, 0, 'First bar progress...')
            %
            %   This example creates a figure with a 3-by-2 grid of waitbars. The
            %   figure is titled 'My wait bars', and the bars are colored cyan. The
            %   Update method sets the upper left bar to a value of zero, displays the
            %   title 'First bar progress...' and makes it visible.
            %
            %   See also: Initialize.
            if nargin > 0
                self.Initialize(varargin{:})
            end
        end
        
        function Initialize(self, rowCount, colCount, figureName, color)
            %Initialize - initialize & display the MultiWaitBar
            %   obj.Initialize(rowCount, colCount) initializes the MultiWaitBar to
            %   display wait bars in a rowCount-by-colCount grid. The bars are not made
            %   visible until they are updated (see Update method).
            %
            %   obj.Initialize(rowCount, colCount, figureName) also adds the specified
            %   figure name.
            %
            %   obj.Initialize(rowCount, colCount, figureName, color) also specifies
            %   the wait bar color, where the color argument is a valid ColorSpec
            %   (default = 'r').
            %
            %   See also: Update.
            if ~exist('color', 'var')
                color = 'r';
            end
            self.rowCount = rowCount;
            self.colCount = colCount;
            self.isVisible = false(self.rowCount, self.colCount);
            self.waitBarAxes = WaitBarAxes();
            for ixRow = rowCount:-1:1
                for ixCol = colCount:-1:1
                    self.waitBarAxes(ixRow, ixCol) = WaitBarAxes();
                end
            end
            
            % set units to pixels
            oldUnits = get(0,'Units');
            set(0, 'Units', 'points');
            screenSize = get(0,'ScreenSize');
            pointsPerPixel = 72/get(0,'ScreenPixelsPerInch');
            
            % compute figure size and position
            self.w = 360 * pointsPerPixel;
            self.h = 75 * pointsPerPixel;
            width = colCount * self.w;
            height = rowCount * self.h;
            pos = [ ...
                screenSize(3)/2-width/2, ...
                min(screenSize(4) - height - 25, screenSize(4)*3/4-height/2), ...
                width, ...
                height];
            
            
            % initialize figure
            if ~exist('figureName', 'var')
                figureName = '';
            end
            self.hFig = figure('Units','points',...
                'Position',         pos,...
                'Resize',           'off',...
                'CreateFcn',        '',...
                'NumberTitle',      'off',...
                'Name',             figureName,...
                'IntegerHandle',    'off',...
                'MenuBar',          'none',...
                'ColorMap',         [],...
                'CloseRequestFcn',  @closeRequest,...
                'Visible',          'on');
            
            % initialize WaitBarAxes
            for ixRow = 1:self.rowCount
                for ixCol = 1:self.colCount
                    axesPosition = [ ...
                        (ixCol - 1 + 0.05) / self.colCount, ...
                        (self.rowCount - ixRow + 0.3) / self.rowCount, ...
                        0.9 / self.colCount, ...
                        0.2 / self.rowCount];
                    self.waitBarAxes(ixRow, ixCol).Initialize( ...
                        axes('Position', axesPosition, 'Visible', 'off'), color); %#ok<LAXES>
                end
            end
            
            % reset units
            set(0, 'Units', oldUnits);
            
            function closeRequest(varargin)
                self.Close();
            end
        end
        
        function waitBarAxes = GetWaitBarAxes(self, row, column)
            %GetWaitBarAxes - get WaitBarAxes object.
            %   waitBarAxes = obj.GetWaitBarAxes(row, column) gets the WaitBarAxes
            %   handle object at the specified position. This handle can be used to
            %   directly manipulate one of the wait bars, and can even be passed to
            %   other functions, objects, etc.
            %
            %   Example:
            %       waitBarAxes = obj.GetWaitBarAxes(row, column);
            %       waitBarAxes.Update(progress, axesTitle);
            %
            %   The above example is equivalent to obj.Update(row, column, progress,
            %   axesTitle).
            waitBarAxes = self.waitBarAxes(row, column);
        end
        
        function Update(self, row, column, varargin)
            %Update - update a bar.
            %   obj.Update(row, column, progress) updates the bar at the specified row
            %   and column and makes it visible. The value of progress must be a number
            %   in the range [0, 1].
            %
            %   obj.Update(row, column, progress, axesTitle) also updates the bar
            %   title.
            %
            %   obj.Update(row, column, progress, axesTitle, color) also updates the bar
            %   color.
            self.waitBarAxes(row, column).Update(varargin{:});
            if ~self.isVisible(row, column)
                self.Show(row, column);
            end
        end
        
        function Show(self, row, column)
            %Show - show a waitbar.
            %   obj.Show(row, column) shows the waitbar at the specified row and column
            %   position.
            self.waitBarAxes(row, column).Show();
        end
        
        function Hide(self, row, column)
            %Hide - hide a waitbar.
            %   obj.Hide(row, column) hides the waitbar at the specified row and column
            %   position.
            self.waitBarAxes(row, column).Hide();
        end
        
        function Close(self)
            %Close - close the wait bar window.
            %   obj.Close() deletes the wait bar window.
            if ~isempty(self.hFig)
                delete(self.hFig)
            end
        end
        
    end % methods
    
end

