#include <iostream>
using namespace std;

int main() {
   /* Type your code here. */
   while(true) {
     string s;
     getline(cin, s);
     
     if(s == "quit") {
        break;
     }
     
     for(unsigned int i = 0; i < s.length()/2; i++) {
        char tmp = s[i];
        s[i] = s[s.length() - 1 - i];
        s[s.length() -1 -i] = tmp;
     }
     
     cout << s << endl;
   }

   return 0;
}
