function vals = my_logspace(start, stop, b, n)

    % Create log-spaced values between start and stop.
    % Works more like linspace than the built in matlab logspace.

    sp = 0.01; 
    % need this if start is 1.
    
    e1 = log(start+sp)/log(b);
    e2 = log(stop+sp)/log(b);

    vals = logspace(e1, e2, n);
    vals = vals - sp;
