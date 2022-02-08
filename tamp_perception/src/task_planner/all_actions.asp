#program step(n).

% Generate actions at each time step
1{
    load(O, n) : object(O);
    unload(O,T,n) : object(O), tableloc(T);
    navigate_to(L, n): location(L)
}1.


#show load/2.
#show unload/3.
#show navigate_to/2.
