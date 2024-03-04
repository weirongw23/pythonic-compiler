use std::rc::Rc;

#[derive(Debug)]
pub struct List<T>(Option<Rc<(T, List<T>)>>);

impl<T> List<T> {
    pub fn push(&mut self, x: T) {
        *self = List(Some(Rc::new((x, (*self).clone()))))
    }

    pub fn iter(&self) -> Self {
        self.clone()
    }

    pub fn empty() -> Self {
        List(None)
    }
    pub fn new() -> Self {
        List(None)
    }
}

impl<T> Clone for List<T> {
    fn clone(&self) -> Self {
        match &self.0 {
            None => List(None),
            Some(x) => List(Some(x.clone())),
        }
    }
}

impl<T> Iterator for List<T>
    where
        T: Copy,
{
    type Item = T;
    fn next(&mut self) -> Option<T> {
        match &self.0 {
            None => None,
            Some(p) => {
                let t = p.0;
                *self = p.1.clone();
                Some(t)
            }
        }
    }
}

impl<T> std::iter::FromIterator<T> for List<T> {
    fn from_iter<S>(iter: S) -> Self
        where
            S: IntoIterator<Item = T>,
    {
        let mut l = List::empty();

        for t in iter {
            l = List(Some(Rc::new((t, l))))
        }
        l
    }
}
