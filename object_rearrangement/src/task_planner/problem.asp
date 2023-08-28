#include <incmode>.

#program base.

location(src).
%location(linit).
tableloc(tinit).
tableloc(t0).
tableloc(t1).
tableloc(t2).
tableloc(t3).
tableloc(t4).
tableloc(t5).

location(l0).
location(l1).

location(l2).
location(l3).

location(l4).
location(l5).

nextto(l0, t0).
nextto(l1, t1).
nextto(l2, t2).
nextto(l3, t3).
nextto(l4, t4).
nextto(l5, t5).
nextto(src, tinit).

object(o0).
object(o1).
object(o2).
object(o3).
object(o4).
object(o5).

% initial state:
init(at(self, src)).
%init(unloadedto(o0,tinit)).
%init(unloadedto(o1,tinit)).
%init(unloadedto(o2,tinit)).
init(inhand(self, o0)).
init(inhand(self, o1)).
init(inhand(self, o2)).
init(inhand(self, o3)).
init(inhand(self, o4)).
init(inhand(self, o5)).

holds(F,0) :- init(F).

% goal state:
%goal(inhand(self, o0)).
%goal(unloadedto(o0, t0)).
%goal(-inhand(self, o0)).
%goal(at(self, l0_top)).
goal(unloadedto(o0,t0)).
goal(unloadedto(o1,t1)).
goal(unloadedto(o2,t2)).
goal(unloadedto(o3,t3)).
goal(unloadedto(o4,t4)).
goal(unloadedto(o5,t5)).

#program check(n).
% Test
:- query(n), goal(F), not holds(F,n).


