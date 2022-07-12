classdef DemoAsynchronousUpdates < handle
    
    properties
        timers;
    end
    
    methods
        
        function self = DemoAsynchronousUpdates()
            % display documentation
            doc WaitBarAxes
            
            % instantiate the MultiWaitBar as a 2-by-2 grid
            mwb = MultiWaitBar(2, 2, 'Asynchronous update demo...');
            
            % create timers that will perform asynchronous updates of the wait bars
            duration = 60;
            self.timers = self.BarTimer(@(o, e)self.Update1(o, e, mwb.GetWaitBarAxes(1, 1)), 0.6, duration / 0.6);
            self.timers(2) = self.BarTimer(@(o, e)self.Update2(o, e, mwb.GetWaitBarAxes(2, 1)), 0.3, duration / 0.3);
            self.timers(3) = self.BarTimer(@(o, e)self.Update3(o, e, mwb.GetWaitBarAxes(1, 2)), 0.2, duration / 0.2);
            self.timers(4) = self.BarTimer(@(o, e)self.Update4(o, e, mwb.GetWaitBarAxes(2, 2)), 0.15, duration / 0.15);
            
            % start the timers then exit.
            start(self.timers);
        end
        
    end % public methods
    
    methods (Static = true)
        
        function tmr = BarTimer(callbackFunction, period, repeatCount)
            tmr = timer(...
                'ExecutionMode',    'fixedRate', ...
                'TimerFcn',         callbackFunction, ...
                'TasksToExecute',   round(repeatCount), ...
                'Period',           period);
        end
        
        % Callback functions for the timers. These are executed asynchronously.
        % Note that the callbacks have no knowledge of which bar they are updating.
        function Update1(~, ~, waitBarAxes)
            persistent ix
            if isempty(ix)
                ix = 0;
            end
            waitBarAxes.Update((1 + mod(ix, 100)) / 100, 'Red bar...', 'r');
            ix = ix + 1;
        end
        
        function Update2(~, ~, waitBarAxes)
            persistent ix
            if isempty(ix)
                ix = 0;
            end
            waitBarAxes.Update((1 + mod(ix, 100)) / 100, 'Blue bar...', 'b');
            ix = ix + 1;
        end
        
        function Update3(~, ~, waitBarAxes)
            persistent ix
            if isempty(ix)
                ix = 0;
            end
            waitBarAxes.Update((1 + mod(ix, 100)) / 100, 'Green bar...', 'g');
            ix = ix + 1;
        end
        
        function Update4(~, ~, waitBarAxes)
            persistent ix
            if isempty(ix)
                ix = 0;
            end
            x = (1 + mod(ix, 100)) / 100;
            waitBarAxes.Update(x, 'Goofy bar...', hsv2rgb([x, 1, 1]));
            ix = ix + 1;
        end
        
    end % public static methods
    
end