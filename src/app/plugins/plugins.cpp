// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#include <map>

#include "./plugins.h"

int stf::Plugin::n_plugins = 0;

std::vector<stf::Plugin> stf::GetPluginLib() {
    std::vector< stf::Plugin > extList;

    // Add a user-defined function:
    extList.push_back(Plugin(wxT("Create bootstrapped datasets"),bootstrap));

    std::vector<wxString> labels;
    Vector_double defaults;
    labels.push_back(wxT("Pulses for IV:"));defaults.push_back(6.0);
    labels.push_back(wxT("Voltage of first:"));defaults.push_back(-80.0);
    labels.push_back(wxT("Voltage step:"));defaults.push_back(-20.0);
    labels.push_back(wxT("First trace:"));defaults.push_back(1.0);
    stf::UserInput ivInput(labels,defaults,wxT("Analyze IV"));

    extList.push_back(Plugin(wxT("Analyze IV"),analyze_iv,ivInput));

    return extList;
}

Recording stf::bootstrap(
        const Recording& data,
        const Vector_double& input,            
        std::map< wxString, double >& results
) {
    if (data.GetSelectedSections().empty()) {
        throw std::runtime_error("Exception in stf::bootstrap()\n"
                "No sections have been selected");
    }
    int nData=100;

    // create a vector of resampled datasets:
    std::vector< std::vector< std::size_t > >
    syntheticIndex(stf::randomPermutation(data.GetSelectedSections(),nData));
    Channel TempChannel(nData);
    // write number of resampled datasets into header for Neuron:
    for (int n=0;n<nData;++n) {
        //initialize temporary sections and channels:
        Section TempSection(data.cur().size()),TempSig(0);
        // no shifting:
        std::vector<int> shift(data.cur().size(),0);

        data.MakeAverage(TempSection,TempSig,data.GetCurCh(),syntheticIndex[n],false,shift);
        try {
            TempChannel.InsertSection(TempSection,n);
        }
        catch (...) {
            throw;
        }
    }

    Recording retRec(TempChannel);
    retRec.CopyAttributes(data);
    return retRec;
}

Recording stf::analyze_iv(
        const Recording& data,
        const Vector_double& input,
        std::map< wxString, double >& results
) {
    if (input.size()==0) {
        return Recording(0);
    }
    // some corrections because user input is 1-based:
    int ivPulses = (int)input[0];
    //	double V_start=input[1];
    double V_step=input[2];
    int traceToBegin=(int)input[3]-1;

    if (ivPulses <= 0 || traceToBegin < 0)
    {
        throw std::runtime_error("Error in stf::analyze_iv():\n"
                "Check settings");
    }

    // Create a new channel:
    Channel TempChannel(ivPulses);
    // Loop:
    for (int m = 0; m < ivPulses; ++m) {
        // A vector of valarrays to get the average:
        std::vector<Vector_double > set;
        for (std::size_t n=traceToBegin+m;
             n < (std::size_t)data.get()[data.GetCurCh()].size();
             n += ivPulses)
        {
            // Add this trace to set:
            set.push_back(data.get()[data.GetCurCh()][n].get());
        }
        // calculate average and create a new section from it:
        try {
            Section TempSection(average(set));
            TempChannel.InsertSection(TempSection,m);
        }
        catch (...) {
            throw;
        }
    }
    Recording Average(TempChannel);
    Average.CopyAttributes(data);
    Average.CopyCursors(data);

    // Calculate input resistance:
    for (int n=0; n < ivPulses; ++n) {
        Average.SetCurCh(0);
        Average.SetCurSec(n);
        try {
            Average.Measure();
        }
        catch (...) {
            throw;
        }
        double I_amp=Average.GetPeak()-Average.GetBase();
        double dV=(n+1)*V_step;
        double R=dV/I_amp;
        wxString strAmp; strAmp << wxT("Amp (") << dV << wxT(" mV)");
        results[strAmp] = I_amp;
        wxString strR; strR << wxT("R (") << dV << wxT(" mV)");
        results[strR] = R;
    }
    // rewind:
    Average.SetCurSec(0);
    return Average;

}

std::vector<std::vector<std::size_t> >
stf::randomPermutation(const std::vector<std::size_t>& input, std::size_t B) {
    // Create a vector with B repetitions of each of input's elements:
    std::vector< std::size_t > superVec(input.size()*B);
    for (std::size_t n=0;n<B;++n) {
        for (std::size_t m=0;m<input.size();++m) {
            superVec[n*input.size()+m]=input[m];
        }
    }

    // perform a random permutation:
    std::random_shuffle(superVec.begin(),superVec.end());

    // cut superVec into B pieces of length input.size():
    std::vector<std::vector<std::size_t> > retVec;
    retVec.reserve(B);
    for (std::size_t n=0;n<B;++n) {
        std::vector<std::size_t> smallVec(input.size());
        for (std::size_t m=0;m<input.size();++m) {
            smallVec[m]=superVec[n*input.size()+m];
        }
        retVec.push_back(smallVec);
    }
    return retVec;
}

Vector_double
stf::average(const std::vector< std::vector< double > >& set) {
    if (set.empty()) {
        throw std::runtime_error("Array of size 0 in stf::average()");
    }
    // find largest valarray within set:
    std::size_t max=0;
    for (std::size_t n=0;n<set.size();++n) {
        if (set[n].size()>max) max=set[n].size();
    }
    if (max==0) {
        throw std::runtime_error("Array of size 0 in stf::average()");
    }
    // valarray for return:
    Vector_double retVa(max);

    for (std::size_t m=0;m<max;++m) {
        double sum=0.0;
        int count=0;
        // sum up at position m if it exists:
        std::vector< std::vector< double > >::const_iterator cit;
        for (cit = set.begin(); cit != set.end(); cit++) {
            if (m < cit->size()) {
                sum += (*cit)[m];
                count++;
            }
        }
        retVa[m] = sum/count;
    }
    return retVa;
}
